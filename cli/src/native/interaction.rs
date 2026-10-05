use std::collections::HashMap;

use serde_json::Value;

use super::cdp::client::CdpClient;
use super::cdp::types::*;
use super::element::{
    resolve_element_center, resolve_element_object_id, session_viewport_offset, RefMap,
};

/// Outcome of a click. `dialog_opened` is true if a JavaScript dialog opened
/// mid-sequence (the page is then blocked until `dialog accept`/`dismiss`).
/// `pending_release` is set only when the dialog opened after mousePressed but
/// before mouseReleased: the button is logically held until the caller
/// dispatches the release (done once the dialog is resolved), otherwise the
/// next click would register as a drag or double-click.
#[derive(Default)]
pub struct ClickResult {
    /// Final pointer position in the top-level page viewport, including dialogs.
    pub position: (f64, f64),
    pub dialog_opened: bool,
    pub pending_release: Option<PendingRelease>,
    pub x: f64,
    pub y: f64,
    pub button_pressed: bool,
}

pub struct PendingRelease {
    pub session_id: String,
    pub x: f64,
    pub y: f64,
    pub button: String,
}

pub async fn click(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    button: &str,
    click_count: i32,
    iframe_sessions: &HashMap<String, String>,
) -> Result<ClickResult, String> {
    let (x, y, effective_session_id) = resolve_element_center(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;
    // A click-triggered dialog can fire on the frame's own session (OOPIF) or
    // on the top-level page session; both count as "ours". A dialog on any
    // other session belongs to a background tab and must not abort this click.
    let offset =
        session_viewport_offset(client, session_id, &effective_session_id, iframe_sessions).await?;
    let mut result = dispatch_click(
        client,
        &effective_session_id,
        &[effective_session_id.as_str(), session_id],
        x,
        y,
        button,
        click_count,
    )
    .await?;
    // Compute before dispatch: a click may navigate or open a blocking dialog.
    result.position = (x + offset.0, y + offset.1);
    (result.x, result.y) = result.position;
    Ok(result)
}

pub async fn dblclick(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<ClickResult, String> {
    click(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        "left",
        2,
        iframe_sessions,
    )
    .await
}

pub async fn hover(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(f64, f64), String> {
    let (x, y, effective_session_id) = resolve_element_center(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;
    let offset =
        session_viewport_offset(client, session_id, &effective_session_id, iframe_sessions).await?;
    client
        .send_command_typed::<_, Value>(
            "Input.dispatchMouseEvent",
            &DispatchMouseEventParams {
                event_type: "mouseMoved".to_string(),
                x,
                y,
                button: None,
                buttons: None,
                click_count: None,
                delta_x: None,
                delta_y: None,
                modifiers: None,
            },
            Some(&effective_session_id),
        )
        .await?;
    Ok((x + offset.0, y + offset.1))
}

/// Input types whose value `fill` sets directly, as Playwright's `fill` does.
/// Chrome drops inserted text on them, or on color and range writes it into
/// whichever field last held the selection.
const SET_VALUE_INPUT_TYPES: &[&str] = &[
    "color",
    "date",
    "time",
    "datetime-local",
    "month",
    "range",
    "week",
];

/// Input types that take typed text.
const TYPE_INPUT_TYPES: &[&str] = &[
    "", "email", "number", "password", "search", "tel", "text", "url",
];

/// The focused element under a document or shadow root, through open shadow
/// roots and same-origin frames. Like every walk here, it stops at a node it
/// has seen: a page's named elements can shadow the properties it reads (a
/// form's controls shadow the form's own, a document's forms and frames the
/// document's), which could lead it back round.
const DEEP_ACTIVE_JS: &str = r#"(root) => {
    let active = root.activeElement;
    const seen = new Set();
    while (active && !seen.has(active)) {
        seen.add(active);
        const inner = (active.shadowRoot && active.shadowRoot.activeElement) ||
            (active.contentDocument && active.contentDocument.activeElement);
        if (!inner) break;
        active = inner;
    }
    return active;
}"#;

/// Whether `n` is `node` or inside it in the composed tree, through same-origin
/// frames: a slotted node steps to its slot, a shadow root to its host and a
/// document to its frame element. Only a shadow root reads `host` and only a
/// document `frameElement`: a document's named properties shadow `host`, so a
/// `<form name="host">` would lead back down into the page. `assignedSlot`
/// hides a slot in a closed shadow root, so failing that, `n` or a node above
/// it may be assigned to `node` itself or a slot inside it, which `node` can
/// see. A form control named parentNode or assignedSlot makes the form's own
/// property lead back down to it; each walk stops at a node it has seen.
const UNDER_JS: &str = r#"(n, node) => {
    const up = (x) => x.nodeType === 11 ? x.host : x.nodeType === 9 ? x.defaultView && x.defaultView.frameElement : x.assignedSlot || x.parentNode;
    for (let x = n, seen = new Set(); x && !seen.has(x); x = up(x)) {
        if (x === node) return true;
        seen.add(x);
    }
    const assigned = new Set();
    for (const slot of [node, ...node.querySelectorAll('slot')]) {
        if (typeof slot.assignedNodes !== 'function') continue;
        for (const a of slot.assignedNodes({ flatten: true })) assigned.add(a);
    }
    for (let x = n, seen = new Set(); x && assigned.size && !seen.has(x); x = up(x)) {
        if (assigned.has(x)) return true;
        seen.add(x);
    }
    return false;
}"#;

/// The focused element as seen from a node. Its own root comes first, since
/// page JS holding the node can read that root even when it is closed. When
/// focus is outside it (on a node slotted in from the host's light DOM, or
/// elsewhere), the enclosing roots up to the document are read in turn.
fn focused_from_js() -> String {
    format!(
        r#"(node) => {{
            for (let root = node.getRootNode(), seen = new Set(); root && !seen.has(root); root = root.nodeType === 11 ? root.host.getRootNode() : null) {{
                seen.add(root);
                const active = ({DEEP_ACTIVE_JS})(root);
                if (active) return active;
            }}
            return null;
        }}"#
    )
}

/// Whether the element may carry a closed shadow root, which page JS can't
/// see: it has no open one, and it is a custom element or a built-in element
/// attachShadow accepts.
const CLOSED_ROOT_POSSIBLE_JS: &str = r#"(el) => !el.shadowRoot && (el.localName.includes('-') ||
    el.matches('article, aside, blockquote, body, div, footer, h1, h2, h3, h4, h5, h6, header, main, nav, p, section, span'))"#;

/// Whether the element is editable: contenteditable, or styled
/// `-webkit-user-modify: read-write` (or read-write-plaintext-only), which
/// Chrome edits, focuses and places a caret in like contenteditable while
/// isContentEditable stays false. The computed value inherits within a tree
/// and Chrome resets it to read-only at a shadow root, as editability.
const EDITABLE_JS: &str = r#"(el) => {
    if (el.isContentEditable) return true;
    const view = el.ownerDocument.defaultView;
    return !!view && /^read-write/.test(view.getComputedStyle(el).getPropertyValue('-webkit-user-modify'));
}"#;

/// Editable content: the editing host of an editable region or content
/// inside it, focusable or not. Chrome reports form controls inside the
/// region as editable too; those aren't editable content.
fn editable_content_js() -> String {
    format!(
        r#"(el) => {{
            const editable = {EDITABLE_JS};
            if (!editable(el)) return false;
            if (!el.parentElement || !editable(el.parentElement)) return true;
            return !el.matches('input, textarea, select, button, output, meter, progress, object, embed, iframe');
        }}"#
    )
}

/// Playwright's follow-label step: an element inside a label stands for the
/// label's control. Controls, editable content (contenteditable, CSS-editable
/// or EditContext) and a frame the page can't read, which takes text itself
/// (see `text_entry`), stand for themselves.
fn label_control_js() -> String {
    format!(
        r#"function() {{
            if (this.matches('a, input, textarea, button, select, [role=link], [role=button], [role=checkbox], [role=radio]') || ({EDITABLE_JS})(this) || this.editContext) return null;
            if (this.matches('iframe, frame') && this.contentDocument === null) return null;
            const label = this.closest('label');
            return label && label.control !== this ? label.control : null;
        }}"#
    )
}

/// How text reaches an element under Playwright's `fill` contract.
#[derive(Debug, PartialEq)]
enum TextEntry {
    /// Set the value directly; holds the input type.
    SetValue(String),
    /// Focus, then type.
    Type,
}

/// Decides how text reaches the element the probe described, or why it
/// can't. `editable` covers contenteditable, CSS-editable
/// (`-webkit-user-modify`) and EditContext editors.
/// `unreadable_frame` marks a frame whose document the page can't read
/// (cross-origin or sandboxed): text goes to whatever has focus inside it.
fn text_entry(
    target: &str,
    tag: &str,
    input_type: &str,
    editable: bool,
    unreadable_frame: bool,
) -> Result<TextEntry, String> {
    match tag {
        "input" if SET_VALUE_INPUT_TYPES.contains(&input_type) => {
            Ok(TextEntry::SetValue(input_type.to_string()))
        }
        "input" if TYPE_INPUT_TYPES.contains(&input_type) => Ok(TextEntry::Type),
        "input" => Err(format!(
            "Input '{}' of type \"{}\" cannot be filled",
            target, input_type
        )),
        "textarea" => Ok(TextEntry::Type),
        _ if editable => Ok(TextEntry::Type),
        "iframe" | "frame" if unreadable_frame => Ok(TextEntry::Type),
        _ => Err(format!(
            "Element '{}' is not an <input>, <textarea> or [contenteditable] element",
            target
        )),
    }
}

/// The value format each set-value input type accepts.
fn input_value_format(input_type: &str) -> &'static str {
    match input_type {
        "color" => "#rrggbb",
        "date" => "YYYY-MM-DD",
        "time" => "HH:MM or HH:MM:SS",
        "datetime-local" => "YYYY-MM-DDTHH:MM",
        "month" => "YYYY-MM",
        "week" => "YYYY-Www, like 2024-W05",
        "range" => "a number within the input's min, max and step",
        _ => "a value the input accepts",
    }
}

fn not_focused_error(target: &str) -> String {
    format!(
        "Element '{}' did not take focus; it may be hidden, disabled or inert",
        target
    )
}

/// Runs `function` on the element. A page exception is an error, not a
/// silent no-op.
async fn call_function_on(
    client: &CdpClient,
    session_id: &str,
    object_id: &str,
    function: String,
    arguments: Option<Vec<CallArgument>>,
    return_by_value: bool,
) -> Result<RemoteObject, String> {
    let result: EvaluateResult = client
        .send_command_typed(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration: function,
                object_id: Some(object_id.to_string()),
                arguments,
                return_by_value: Some(return_by_value),
                await_promise: Some(false),
            },
            Some(session_id),
        )
        .await?;
    if let Some(details) = &result.exception_details {
        let message = details
            .exception
            .as_ref()
            .and_then(|e| e.description.as_deref())
            .unwrap_or(&details.text);
        return Err(format!("Evaluation error: {}", message));
    }
    Ok(result.result)
}

/// Runs `function` on the element and returns its result by value.
async fn call_on_element(
    client: &CdpClient,
    session_id: &str,
    object_id: &str,
    function: String,
    arguments: Option<Vec<CallArgument>>,
) -> Result<Value, String> {
    let result = call_function_on(client, session_id, object_id, function, arguments, true).await?;
    Ok(result.value.unwrap_or(Value::Null))
}

/// Runs `function` on the element and returns the element it returns, if any.
async fn call_for_element(
    client: &CdpClient,
    session_id: &str,
    object_id: &str,
    function: String,
) -> Result<Option<String>, String> {
    let result = call_function_on(client, session_id, object_id, function, None, false).await?;
    Ok(result.object_id)
}

/// The element's closed shadow root. Page JS can't reach one; CDP can. An
/// engine that can't answer these DOM commands (partial CDP) is taken to have
/// none, so fill and type still work there.
async fn closed_shadow_root(
    client: &CdpClient,
    session_id: &str,
    object_id: &str,
) -> Option<String> {
    let node = client
        .send_command(
            "DOM.describeNode",
            Some(serde_json::json!({ "objectId": object_id, "depth": 0 })),
            Some(session_id),
        )
        .await
        .ok()?;
    let backend_node_id = node
        .pointer("/node/shadowRoots")
        .and_then(Value::as_array)
        .and_then(|roots| roots.iter().find(|r| r["shadowRootType"] == "closed"))
        .and_then(|root| root["backendNodeId"].as_i64())?;
    let root: DomResolveNodeResult = client
        .send_command_typed(
            "DOM.resolveNode",
            &DomResolveNodeParams {
                backend_node_id: Some(backend_node_id),
                node_id: None,
                object_group: Some("agent-browser".to_string()),
            },
            Some(session_id),
        )
        .await
        .ok()?;
    root.object.object_id
}

/// What the focus call did.
pub(super) enum FocusCall {
    /// Nothing yet: focus is already inside the host, so whether the host can
    /// take focus is needed first. Holds what that rests on, as page JS reads
    /// it, to tell whether the host changed before the call that uses it.
    Inside(String),
    /// Focused the host; holds what the call saw, for `settle_focus`.
    Done(Option<String>),
}

/// Focuses the host and returns what the call saw, for `settle_focus` to read
/// in a later call. Whether the host passes focus on is decided from the call
/// alone: focus that a timer or anything else moves afterwards says nothing
/// about the host. With focus already inside, whether the host can take focus
/// is read first, in an extra call. With `refocus`, the host is blurred first,
/// so focus it already holds is taken again.
pub(super) async fn focus_host(
    client: &CdpClient,
    session_id: &str,
    host: &str,
    refocus: bool,
) -> Result<Option<String>, String> {
    let inputs = match focus_call(client, session_id, host, refocus, None).await? {
        FocusCall::Done(call) => return Ok(call),
        FocusCall::Inside(inputs) => inputs,
    };
    let takes_focus = focusable(client, session_id, host).await;
    match focus_call(
        client,
        session_id,
        host,
        refocus,
        Some((inputs, takes_focus)),
    )
    .await?
    {
        FocusCall::Done(call) => Ok(call),
        // Given whether the host takes focus, the call always focuses it.
        FocusCall::Inside(_) => Ok(None),
    }
}

/// One focus call. Without `focusable`, it stops short when focus is already
/// inside the host; with it, it counts the accessibility answer only if
/// nothing it rests on has changed since.
pub(super) async fn focus_call(
    client: &CdpClient,
    session_id: &str,
    host: &str,
    refocus: bool,
    focusable: Option<(String, bool)>,
) -> Result<FocusCall, String> {
    let focused_from = focused_from_js();
    let focus = format!(
        r#"function(refocus, focusable, inputs) {{
            const root = this.getRootNode();
            const view = this.ownerDocument.defaultView;
            const deepActive = () => ({focused_from})(this);
            const under = (node) => ({UNDER_JS})(node, this);
            const before = deepActive();
            // The host hands focus on by delegatesFocus, its own focus() or, as
            // a same-origin frame, to its document. Its focus() is native when
            // the object that holds it is the HTMLElement, SVGElement or
            // MathMLElement prototype of whichever realm made the element: one
            // made in a frame keeps that frame's prototypes after it moves
            // into this document. An interface prototype carries its name as
            // its own toStringTag; this realm's are known by identity too.
            let owner = this;
            while (owner && !Object.prototype.hasOwnProperty.call(owner, 'focus')) owner = Object.getPrototypeOf(owner);
            const tag = owner && Object.getOwnPropertyDescriptor(owner, Symbol.toStringTag);
            const held = owner && Object.getOwnPropertyDescriptor(owner, 'focus');
            const nativeFocus = !!held && 'value' in held &&
                ([view.HTMLElement, view.SVGElement, view.MathMLElement].some((type) => type && type.prototype === owner) ||
                    ['HTMLElement', 'SVGElement', 'MathMLElement'].includes(tag && tag.value));
            const handsOn = !!this.contentDocument || !!(this.shadowRoot && this.shadowRoot.delegatesFocus) || !nativeFocus;
            // What focus() depends on that page JS can read: a tabindex, what
            // HTML makes focusable without one (a link or area with an href, a
            // details element's first summary, an input that isn't hidden,
            // media with controls, an editing host, contenteditable or CSS),
            // not being disabled by itself or a fieldset, nothing making it
            // inert (the attribute or CSS interactivity on it or above it,
            // through frames, or a modal dialog open in its document or one
            // above) and a box.
            const editable = {EDITABLE_JS};
            const focusInputs = () => {{
                let inert = false;
                let modals = 0;
                let styled = true;
                const seen = new Set();
                for (let x = this; x && !seen.has(x); x = x.nodeType === 11 ? x.host : x.nodeType === 9 ? x.defaultView && x.defaultView.frameElement : x.assignedSlot || x.parentNode) {{
                    seen.add(x);
                    if (x.nodeType === 9) {{
                        // An engine without :modal has no modal dialogs to count.
                        try {{
                            modals += x.querySelectorAll(':modal').length;
                        }} catch {{}}
                        styled = true;
                    }} else if (x.nodeType === 1) {{
                        inert = inert || x.inert === true;
                        // CSS interactivity inherits within a document, so the
                        // host's and each frame element's cover the rest.
                        if (styled) inert = inert || x.ownerDocument.defaultView.getComputedStyle(x).interactivity === 'inert';
                        styled = false;
                    }}
                }}
                const details = this.localName === 'summary' && this.parentElement;
                const firstSummary = !!details && details.localName === 'details' &&
                    [...details.children].find((child) => child.localName === 'summary') === this;
                const rendered = !this.checkVisibility || this.checkVisibility({{ visibilityProperty: true }});
                return JSON.stringify([
                    this.getAttribute('tabindex'),
                    this.hasAttribute('href') || this.hasAttribute('xlink:href'),
                    firstSummary,
                    this.localName === 'input' && this.type === 'hidden',
                    (this.localName === 'audio' || this.localName === 'video') && this.controls,
                    this.isContentEditable,
                    view.getComputedStyle(this).getPropertyValue('-webkit-user-modify'),
                    editable(this) && !(this.parentElement && editable(this.parentElement)),
                    this.matches(':disabled'),
                    inert,
                    modals,
                    rendered,
                ]);
            }};
            // Focus already inside, not on the host, ends where it was if the
            // host's focus() does nothing. Whether the host can take focus
            // tells those apart, read before focusing, since a focus handler
            // may change it.
            if (typeof focusable !== 'boolean' && !handsOn && before !== this && under(before)) return focusInputs();
            // The answer is stale if the host changed since it was read.
            const canTakeFocus = focusable === true && inputs === focusInputs();
            const state = {{ host: this, forwards: handsOn }};
            const fromShadowTree = !!(this.shadowRoot && this.shadowRoot.activeElement);
            // Capture listeners run before the host's own focus handlers, which
            // may stop the event and then pass focus on. The window's runs
            // before any the page put on the document; the root's hears focus
            // move within a shadow tree, which never reaches the window.
            let tookFocus = false;
            const onFocus = () => {{ tookFocus = tookFocus || under(deepActive()); }};
            // The host took focus if its focus() moved focus, fired a focus
            // event under the host or made focus leave the element holding it
            // (a listener the page added first can stop the focus event, then
            // pass focus back). Where focus is when focus() returns doesn't
            // matter: a handler may blur the host and pass focus on in a
            // microtask, so the read checks where it ends up. Failing those,
            // focus inside that ends where it was came back through the host
            // if the host can take focus: its focus() moved focus onto it,
            // whatever stops the events. Focus already on the host goes
            // nowhere.
            const focusHost = () => {{
                const prior = deepActive();
                let left = false;
                const onLeave = () => {{ left = true; }};
                if (prior) {{
                    prior.addEventListener('blur', onLeave, true);
                    prior.addEventListener('focusout', onLeave, true);
                }}
                try {{
                    this.focus();
                }} finally {{
                    if (prior) {{
                        prior.removeEventListener('blur', onLeave, true);
                        prior.removeEventListener('focusout', onLeave, true);
                    }}
                }}
                const took = deepActive() !== prior || tookFocus || left || (canTakeFocus && prior !== this);
                state.forwards = state.forwards || took;
            }};
            view.addEventListener('focus', onFocus, true);
            root.addEventListener('focus', onFocus, true);
            try {{
                if (refocus) this.blur();
                focusHost();
                // Chrome fires no focus event when focus moves from inside the
                // host's shadow tree onto the host, so its focus handlers didn't
                // run; focus it again from outside.
                if (deepActive() === this && fromShadowTree) {{
                    this.blur();
                    focusHost();
                }}
            }} finally {{
                view.removeEventListener('focus', onFocus, true);
                root.removeEventListener('focus', onFocus, true);
            }}
            return state;
        }}"#
    );
    let (inputs, focusable) = match focusable {
        Some((inputs, focusable)) => (Value::from(inputs), Value::Bool(focusable)),
        None => (Value::Null, Value::Null),
    };
    let arguments = [Value::Bool(refocus), focusable, inputs]
        .into_iter()
        .map(|value| CallArgument {
            value: Some(value),
            object_id: None,
        })
        .collect();
    let state = call_function_on(client, session_id, host, focus, Some(arguments), false).await?;
    Ok(match state.value {
        Some(Value::String(inputs)) => FocusCall::Inside(inputs),
        _ => FocusCall::Done(state.object_id),
    })
}

/// Where focus settled after the host's focus call.
pub(super) enum Settled {
    /// On an element inside the host, which stands for the host.
    Inside(String),
    /// On the host itself, pulled out of its closed shadow root, which fires
    /// no focus event, so the host's handlers didn't run.
    PulledOut,
    /// Nothing went through the host: it kept focus itself, or it is a
    /// container that didn't pass focus on. A label around it may stand in.
    Kept,
    /// The host took focus and passed it out of itself. The host stays the
    /// target unless focus went to the control of a label around it, as a
    /// label's own focus() does: the text never lands anywhere else outside.
    Out,
    /// The host passed focus inside, but it moved on before it could be
    /// followed, so nothing is filled.
    Lost,
}

/// Reads where focus went after `focus_host` and follows it. Every microtask
/// the focus queued, however deep its promise chain, has run before the next
/// CDP command does, with no page timer involved.
pub(super) async fn settle_focus(
    client: &CdpClient,
    session_id: &str,
    host: &str,
    call: Option<String>,
) -> Result<Settled, String> {
    let read = read_focus(client, session_id, call).await?;
    settle_read(client, session_id, host, read).await
}

/// The first read after the focus call: whether the call passed focus on and
/// where focus is now, on the host, inside it or outside.
pub(super) async fn read_focus(
    client: &CdpClient,
    session_id: &str,
    call: Option<String>,
) -> Result<Value, String> {
    let Some(call) = call else {
        return Ok(Value::Null);
    };
    let focused_from = focused_from_js();
    let read = format!(
        r#"function() {{
            const {{ host, forwards }} = this;
            const active = ({focused_from})(host);
            const place = !({UNDER_JS})(active, host) ? 'outside' : active === host ? 'host' : 'inside';
            return {{ forwards, place, closedRoot: place !== 'outside' && ({CLOSED_ROOT_POSSIBLE_JS})(active) }};
        }}"#
    );
    call_on_element(client, session_id, &call, read, None).await
}

/// Follows focus from the first read. Page code can move focus before the
/// next read: if it moved onto the host, where it may be inside the host's
/// closed shadow root, that is classified again; if it can't be found inside
/// the host any more, nothing is filled.
pub(super) async fn settle_read(
    client: &CdpClient,
    session_id: &str,
    host: &str,
    read: Value,
) -> Result<Settled, String> {
    let Some(forwards) = read.get("forwards").and_then(Value::as_bool) else {
        return Ok(Settled::Kept);
    };
    let closed_root = read.get("closedRoot") == Some(&Value::Bool(true));
    match read.get("place").and_then(Value::as_str) {
        // Focus inside that the host didn't pass on came from earlier or
        // from elsewhere: the host is just a container.
        Some("inside") if !forwards => Ok(Settled::Kept),
        Some("inside") => {
            // Read from the host again: focus may have moved on since, out of
            // it or into its closed shadow root, where it reads as on the host.
            let focused_from = focused_from_js();
            let inside = format!(
                r#"function() {{
                    const active = ({focused_from})(this);
                    if (active === this) return 'host';
                    return ({UNDER_JS})(active, this) ? active : null;
                }}"#
            );
            let found = call_function_on(client, session_id, host, inside, None, false).await?;
            if found.value == Some(Value::from("host")) {
                return on_host(client, session_id, host, forwards, true).await;
            }
            match found.object_id {
                Some(active) => follow_closed_roots(client, session_id, active, closed_root)
                    .await
                    .map(Settled::Inside),
                None => Ok(Settled::Lost),
            }
        }
        // Focus on the host itself only hides a target behind a closed
        // shadow root.
        Some("host") if closed_root => on_host(client, session_id, host, forwards, false).await,
        Some("host") => Ok(Settled::Kept),
        _ if forwards => Ok(Settled::Out),
        _ => Ok(Settled::Kept),
    }
}

/// Focus reads as on the host, which may mean inside its closed shadow root.
/// `moved` says an earlier read found it elsewhere inside the host: if it
/// isn't in the closed root, where it went is unknown.
async fn on_host(
    client: &CdpClient,
    session_id: &str,
    host: &str,
    forwards: bool,
    moved: bool,
) -> Result<Settled, String> {
    let Some(root) = closed_shadow_root(client, session_id, host).await else {
        return Ok(if moved { Settled::Lost } else { Settled::Kept });
    };
    let hands_on = forwards
        || call_on_element(
            client,
            session_id,
            &root,
            "function() { return this.delegatesFocus; }".to_string(),
            None,
        )
        .await?
            == Value::Bool(true);
    let deep_active = format!("function() {{ return ({DEEP_ACTIVE_JS})(this); }}");
    match call_for_element(client, session_id, &root, deep_active).await? {
        Some(inner) if hands_on => follow_closed_roots(client, session_id, inner, true)
            .await
            .map(Settled::Inside),
        Some(_) => Ok(Settled::Kept),
        None if !hands_on => Ok(Settled::PulledOut),
        None if moved => Ok(Settled::Lost),
        None if !forwards => Ok(Settled::Kept),
        // The host took focus and holds it itself, unless focus left since
        // the first read.
        None => {
            let focused_from = focused_from_js();
            let held = format!("function() {{ return ({focused_from})(this) === this; }}");
            let held = call_on_element(client, session_id, host, held, None).await?;
            Ok(if held == Value::Bool(true) {
                Settled::Kept
            } else {
                Settled::Lost
            })
        }
    }
}

/// Focuses the element and reports where focus settled, following it through
/// same-origin frames and closed shadow roots too.
async fn focused_inside(
    client: &CdpClient,
    session_id: &str,
    host: &str,
) -> Result<Settled, String> {
    let mut refocus = false;
    loop {
        let call = focus_host(client, session_id, host, refocus).await?;
        match settle_focus(client, session_id, host, call).await? {
            // Focus it again from outside, once.
            Settled::PulledOut if !refocus => refocus = true,
            Settled::PulledOut => return Ok(Settled::Kept),
            settled => return Ok(settled),
        }
    }
}

fn lost_focus_error(target: &str) -> String {
    format!(
        "Element '{}' passed focus on, but focus moved again before it could be followed",
        target
    )
}

fn changed_error(target: &str) -> String {
    format!(
        "Element '{}' changed before the text reached it; it may no longer take text",
        target
    )
}

/// Whether the accessibility tree reports the element focusable, as focus()
/// finds it: true for an element with tabindex or one that is focusable by
/// nature, false for a plain container or a disabled, inert, hidden or
/// display: contents element. Pages can't hide it. An engine without the
/// accessibility domain reports it isn't.
pub(super) async fn focusable(client: &CdpClient, session_id: &str, object_id: &str) -> bool {
    let Ok(tree) = client
        .send_command(
            "Accessibility.getPartialAXTree",
            Some(serde_json::json!({ "objectId": object_id, "fetchRelatives": false })),
            Some(session_id),
        )
        .await
    else {
        return false;
    };
    tree.pointer("/nodes/0/properties")
        .and_then(Value::as_array)
        .is_some_and(|properties| {
            properties
                .iter()
                .any(|p| p["name"] == "focusable" && p["value"]["value"] == true)
        })
}

/// Follows focus from `active` into its closed shadow root and any closed
/// roots nested inside; `may_carry_one` says whether `active` can carry one.
/// Handles from DOM.resolveNode live in their frame's own context, which may
/// not be the host's (after `frame`, the host comes through the parent
/// document), and CDP won't mix contexts in one call. So whatever a closed root
/// yields is inside it by construction and is not compared with the host.
async fn follow_closed_roots(
    client: &CdpClient,
    session_id: &str,
    mut active: String,
    may_carry_one: bool,
) -> Result<String, String> {
    if !may_carry_one {
        return Ok(active);
    }
    let deep_active = format!("function() {{ return ({DEEP_ACTIVE_JS})(this); }}");
    while let Some(root) = closed_shadow_root(client, session_id, &active).await {
        match call_for_element(client, session_id, &root, deep_active.clone()).await? {
            Some(inner) => active = inner,
            None => break,
        }
    }
    Ok(active)
}

/// Resolves the element that receives the text. An element that takes focus
/// into itself stands for the element that took it, since typed text goes
/// wherever focus is. Otherwise, as in Playwright, an element inside a label
/// stands for the label's control. Focus goes first because a form-associated
/// custom element can be a label's control itself. The label rule is for an
/// element nothing passed focus on through, or one that passed it to that
/// control (a label's own focus() does): one that passed focus anywhere else
/// stays the target, and one whose focus can't be followed is refused, so the
/// text never lands in the label's control instead.
async fn resolve_text_target(
    client: &CdpClient,
    session_id: &str,
    object_id: String,
    target: &str,
) -> Result<String, String> {
    let settled = focused_inside(client, session_id, &object_id).await?;
    match settled {
        Settled::Inside(inner) => return Ok(inner),
        Settled::Lost => return Err(lost_focus_error(target)),
        Settled::Kept | Settled::PulledOut | Settled::Out => {}
    }
    let Some(control) =
        call_for_element(client, session_id, &object_id, label_control_js()).await?
    else {
        return Ok(object_id);
    };
    if matches!(settled, Settled::Out) {
        let focused_from = focused_from_js();
        let holds = format!("function() {{ return ({UNDER_JS})(({focused_from})(this), this); }}");
        if call_on_element(client, session_id, &control, holds, None).await? != Value::Bool(true) {
            return Ok(object_id);
        }
    }
    match focused_inside(client, session_id, &control).await? {
        Settled::Inside(inner) => Ok(inner),
        Settled::Lost => Err(lost_focus_error(target)),
        _ => Ok(control),
    }
}

/// Resolves the element that receives the text and classifies it before
/// anything is cleared, typed or set, so a refused element is left as it was.
/// Returns what the classification read too, for the typing step to check
/// the element hasn't changed since.
async fn classify_text_target(
    client: &CdpClient,
    session_id: &str,
    object_id: String,
    target: &str,
) -> Result<(String, TextEntry, Value), String> {
    let object_id = resolve_text_target(client, session_id, object_id, target).await?;
    let text_facts = text_facts_js();
    let facts = call_on_element(
        client,
        session_id,
        &object_id,
        format!("function() {{ return ({text_facts})(this); }}"),
        None,
    )
    .await?;
    let entry = text_entry(
        target,
        facts.get("tag").and_then(Value::as_str).unwrap_or_default(),
        facts
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        facts
            .get("editable")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        facts
            .get("unreadableFrame")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    )?;
    Ok((object_id, entry, facts))
}

/// What `text_entry` decides on, read from an element.
fn text_facts_js() -> String {
    let editable_content = editable_content_js();
    format!(
        r#"(el) => ({{
            tag: el.localName,
            type: el.localName === 'input' ? el.type.toLowerCase() : '',
            editable: ({editable_content})(el) || !!el.editContext,
            // A frame's contentDocument is null when the page can't read it.
            unreadableFrame: el.contentDocument === null,
        }})"#
    )
}

/// Makes the page behave as focused, as Playwright does for every page. In a
/// background tab Chrome holds focus events until the page gains focus, so a
/// handler that passes focus on wouldn't run before focus is checked. A
/// cross-origin frame has its own session and renderer, which page focus
/// doesn't reach, so it is set there too. It stays on for the session:
/// turning it off fires blur on the focused element. An engine without it is
/// left as it is.
async fn emulate_page_focus(client: &CdpClient, session_id: &str, frame_session_id: &str) {
    let sessions: &[&str] = if frame_session_id == session_id {
        &[session_id]
    } else {
        &[session_id, frame_session_id]
    };
    for session in sessions {
        let _ = client
            .send_command(
                "Emulation.setFocusEmulationEnabled",
                Some(serde_json::json!({ "enabled": true })),
                Some(session),
            )
            .await;
    }
}

/// Focuses the element and optionally clears it. Typed text goes to whatever
/// has focus in the page's focused frame, so an element that doesn't get
/// focus is an error rather than a write into another field. `facts` is what
/// classification read; a focus handler that changed the element since (made
/// an input a checkbox, say) makes it an error too.
async fn focus_for_typing(
    client: &CdpClient,
    session_id: &str,
    object_id: &str,
    target: &str,
    clear: bool,
    facts: &Value,
) -> Result<(), String> {
    let editable_content = editable_content_js();
    let text_facts = text_facts_js();
    let focus = format!(
        r#"function() {{
            // Focus the element itself. Editable content that can't take focus
            // falls back to its editing host (the body in a designMode
            // document); a control or widget there is refused instead. The
            // climb reads parentElement from the prototype: a form control
            // named parentElement would otherwise lead it round between the
            // form and that control.
            let el = this;
            el.focus();
            const content = ({editable_content})(el);
            if (el.getRootNode().activeElement !== el && content) {{
                const parentOf = Object.getOwnPropertyDescriptor(Node.prototype, 'parentElement').get;
                if (el.ownerDocument.designMode === 'on') el = el.ownerDocument.body;
                else for (let parent = parentOf.call(el); parent && ({EDITABLE_JS})(parent); parent = parentOf.call(el)) el = parent;
                el.focus();
            }}
            return {{ target: this, el, content }};
        }}"#
    );
    let state = call_function_on(client, session_id, object_id, focus, None, false).await?;
    let Some(state) = state.object_id else {
        return Err(not_focused_error(target));
    };
    // Checked in a later call, once the microtasks a focus handler queued have
    // run: one may move focus on.
    let focused = call_on_element(
        client,
        session_id,
        &state,
        format!(
            r#"function(clear, expected) {{
                const {{ target, el, content }} = this;
                // A focus handler may have changed the target since it was
                // classified; typed text must not go to what it is now.
                const now = ({text_facts})(target);
                if (Object.keys(now).some((key) => now[key] !== expected[key])) return 'changed';
                // A host with no box (display: contents) can't take focus, but a
                // caret already inside it gets typed text while nothing else holds
                // focus. The selection is read through the element's root, since
                // document.getSelection() reports a caret inside a shadow tree
                // at the host, unless that holds no range: then the caret is
                // outside the shadow tree, maybe in light-DOM text slotted into
                // the element, which the composed tree counts as inside it.
                const root = el.getRootNode();
                const active = root.activeElement;
                const rootSelection = root.getSelection && root.getSelection();
                const selection = rootSelection && rootSelection.rangeCount > 0 ? rootSelection : el.ownerDocument.getSelection();
                const selectionInside = () => !!selection && selection.rangeCount > 0 &&
                    ({UNDER_JS})(selection.anchorNode, el) && ({UNDER_JS})(selection.focusNode, el);
                const focused = ({DEEP_ACTIVE_JS})(el.ownerDocument);
                const focusAbove = !focused || ({UNDER_JS})(el, focused);
                const caretInside = content && selectionInside() && focusAbove;
                if (active !== el && !caretInside) return false;
                // A frame's body is its activeElement even when the frame lacks
                // focus, so check each frame up the chain holds it. A cross-origin
                // parent can't be read from here.
                for (let w = el.ownerDocument.defaultView; w && w !== w.parent && w.frameElement; w = w.parent) {{
                    if (w.frameElement.getRootNode().activeElement !== w.frameElement) return false;
                }}
                // Focus on a focusable element inside an editable region (a
                // nested editor, a span with tabindex) leaves the caret where it
                // was, maybe in another field, so put it at the end of the
                // element. An editing host places the caret itself on focus.
                const inRegion = !!el.parentElement && ({EDITABLE_JS})(el.parentElement);
                if (active === el && content && inRegion && selection && !selectionInside()) {{
                    selection.collapse(el, el.childNodes.length);
                }}
                // Only inputs and textareas are cleared; `value = ''` elsewhere
                // sets an attribute (an <li> gets value="0").
                if (clear && (target.localName === 'input' || target.localName === 'textarea')) {{
                    target.select();
                    target.value = '';
                    target.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
                return true;
            }}"#
        ),
        Some(vec![
            CallArgument {
                value: Some(serde_json::json!(clear)),
                object_id: None,
            },
            CallArgument {
                value: Some(facts.clone()),
                object_id: None,
            },
        ]),
    )
    .await?;
    match focused {
        Value::Bool(true) => Ok(()),
        Value::String(outcome) if outcome == "changed" => Err(changed_error(target)),
        _ => Err(not_focused_error(target)),
    }
}

/// Sets an input's value directly, then dispatches `input` and `change`. A
/// readonly input is refused, and a value the browser rejects is restored
/// and reported instead.
async fn set_input_value(
    client: &CdpClient,
    session_id: &str,
    object_id: &str,
    target: &str,
    input_type: &str,
    value: &str,
) -> Result<(), String> {
    let value = value.trim();
    // Chrome stores colors lowercased, which would otherwise read back as a mismatch.
    let value = if input_type == "color" {
        value.to_lowercase()
    } else {
        value.to_string()
    };
    let outcome = call_on_element(
        client,
        session_id,
        object_id,
        r#"function(value, type) {
            // The value setter ignores readonly, so refuse it up front.
            if (this.readOnly) return 'readonly';
            this.focus();
            if (this.getRootNode().activeElement !== this) return 'unfocused';
            // A focus handler may have changed the type since it was read.
            if (this.type !== type) return 'changed';
            // The prototype setter bypasses the instance-level value
            // tracking React uses, so React sees the change.
            const { set } = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
            const previous = this.value;
            set.call(this, value);
            // Chrome normalizes some valid values (2024-01-15T09:30:00 to
            // 2024-01-15T09:30, 75.0 to 75), empties date-like values it can't
            // parse and clamps range values to min, max and step. A range value
            // it can't parse becomes the midpoint, which Number() can match
            // (0x32 or +50 for 50), so the text must be a valid floating-point
            // number as HTML defines it first.
            const float = /^-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?$/;
            const accepted = this.type === 'color' ? this.value === value
                : this.type === 'range' ? float.test(value) && Number(this.value) === Number(value)
                : this.value !== '' || value === '';
            if (!accepted) {
                set.call(this, previous);
                return 'malformed';
            }
            this.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
            this.dispatchEvent(new Event('change', { bubbles: true }));
            return 'set';
        }"#
        .to_string(),
        Some(vec![
            CallArgument {
                value: Some(serde_json::json!(value)),
                object_id: None,
            },
            CallArgument {
                value: Some(serde_json::json!(input_type)),
                object_id: None,
            },
        ]),
    )
    .await?;
    match outcome.as_str() {
        Some("set") => Ok(()),
        Some("changed") => Err(changed_error(target)),
        Some("readonly") => Err(format!(
            "Element '{}' is readonly and cannot be filled",
            target
        )),
        Some("malformed") => Err(format!(
            "Malformed value \"{}\" for input '{}' of type \"{}\"; expected {}",
            value,
            target,
            input_type,
            input_value_format(input_type)
        )),
        _ => Err(not_focused_error(target)),
    }
}

/// Fills an element following Playwright's `fill` contract. Text inputs and
/// textareas are cleared and typed into; editable content is typed into at
/// its caret, and a frame the page can't read wherever focus is inside it.
/// Date, time, color and range inputs get their value set directly, because
/// the browser ignores typed text there. Anything else is an error, so text
/// never lands in another field.
pub async fn fill(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    value: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;
    emulate_page_focus(client, session_id, &effective_session_id).await;
    let (object_id, entry, facts) =
        classify_text_target(client, &effective_session_id, object_id, selector_or_ref).await?;

    match entry {
        TextEntry::SetValue(input_type) => {
            set_input_value(
                client,
                &effective_session_id,
                &object_id,
                selector_or_ref,
                &input_type,
                value,
            )
            .await
        }
        TextEntry::Type => {
            focus_for_typing(
                client,
                &effective_session_id,
                &object_id,
                selector_or_ref,
                true,
                &facts,
            )
            .await?;

            // Insert text (keyboard input dispatched at page level, use parent session_id)
            client
                .send_command_typed::<_, Value>(
                    "Input.insertText",
                    &InsertTextParams {
                        text: value.to_string(),
                    },
                    Some(session_id),
                )
                .await?;

            Ok(())
        }
    }
}

/// Types into an element that takes text: a text input, textarea, editable
/// content or a frame the page can't read. Errors on anything else, including
/// date, time, color and range inputs, which only `fill` can set.
#[allow(clippy::too_many_arguments)]
pub async fn type_text(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    text: &str,
    clear: bool,
    delay_ms: Option<u64>,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;
    emulate_page_focus(client, session_id, &effective_session_id).await;
    // Refuse the same elements `fill` does, so text never lands in another field.
    let (object_id, entry, facts) =
        classify_text_target(client, &effective_session_id, object_id, selector_or_ref).await?;
    if let TextEntry::SetValue(input_type) = entry {
        return Err(format!(
            "Input '{}' of type \"{}\" does not take typed text; use fill to set its value",
            selector_or_ref, input_type
        ));
    }
    focus_for_typing(
        client,
        &effective_session_id,
        &object_id,
        selector_or_ref,
        clear,
        &facts,
    )
    .await?;

    type_text_into_active_context(client, session_id, text, delay_ms).await
}

pub async fn type_text_into_active_context(
    client: &CdpClient,
    session_id: &str,
    text: &str,
    delay_ms: Option<u64>,
) -> Result<(), String> {
    let delay = delay_ms.unwrap_or(0);

    for ch in text.chars() {
        if matches!(ch, '\n' | '\r' | '\t') {
            let (key, code, key_code) = char_to_key_info(ch);
            let text_str = key_text(&key);
            client
                .send_command_typed::<_, Value>(
                    "Input.dispatchKeyEvent",
                    &DispatchKeyEventParams {
                        event_type: "keyDown".to_string(),
                        key: Some(key.clone()),
                        code: Some(code.clone()),
                        text: text_str.clone(),
                        unmodified_text: text_str,
                        windows_virtual_key_code: Some(key_code),
                        native_virtual_key_code: Some(key_code),
                        modifiers: None,
                    },
                    Some(session_id),
                )
                .await?;

            client
                .send_command_typed::<_, Value>(
                    "Input.dispatchKeyEvent",
                    &DispatchKeyEventParams {
                        event_type: "keyUp".to_string(),
                        key: Some(key),
                        code: Some(code),
                        text: None,
                        unmodified_text: None,
                        windows_virtual_key_code: Some(key_code),
                        native_virtual_key_code: Some(key_code),
                        modifiers: None,
                    },
                    Some(session_id),
                )
                .await?;
        } else {
            // VS Code/Electron webviews reject repeated dispatchKeyEvent calls
            // carrying printable `text`. Insert printable characters directly
            // and reserve key events for controls like Enter and Tab.
            client
                .send_command_typed::<_, Value>(
                    "Input.insertText",
                    &InsertTextParams {
                        text: ch.to_string(),
                    },
                    Some(session_id),
                )
                .await?;
        }

        if delay > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
        }
    }

    Ok(())
}

pub async fn press_key(client: &CdpClient, session_id: &str, key: &str) -> Result<(), String> {
    press_key_with_modifiers(client, session_id, key, None).await
}

/// Dispatch a keyDown+keyUp sequence for `key` with an optional CDP modifier bitmask.
///
/// Modifier values follow the CDP `Input.dispatchKeyEvent` spec:
/// 1 = Alt, 2 = Control, 4 = Meta (Cmd), 8 = Shift.
///
/// Callers that need a platform-appropriate modifier (e.g. Cmd on macOS,
/// Ctrl elsewhere) must choose the value themselves -- see `cfg!(target_os)`.
pub async fn press_key_with_modifiers(
    client: &CdpClient,
    session_id: &str,
    key: &str,
    modifiers: Option<i32>,
) -> Result<(), String> {
    let (key_name, code, key_code) = named_key_info(key);

    // Suppress text insertion when Control (2) or Meta (4) modifiers are active,
    // since these are command chords (e.g. Ctrl+A = select-all), not text input.
    let has_command_modifier = modifiers.is_some_and(|m| m & (2 | 4) != 0);
    let text = if has_command_modifier {
        None
    } else {
        key_text(&key_name)
    };

    client
        .send_command_typed::<_, Value>(
            "Input.dispatchKeyEvent",
            &DispatchKeyEventParams {
                event_type: "keyDown".to_string(),
                key: Some(key_name.clone()),
                code: Some(code.clone()),
                text: text.clone(),
                unmodified_text: text.clone(),
                windows_virtual_key_code: Some(key_code),
                native_virtual_key_code: Some(key_code),
                modifiers,
            },
            Some(session_id),
        )
        .await?;

    client
        .send_command_typed::<_, Value>(
            "Input.dispatchKeyEvent",
            &DispatchKeyEventParams {
                event_type: "keyUp".to_string(),
                key: Some(key_name),
                code: Some(code),
                text: None,
                unmodified_text: None,
                windows_virtual_key_code: Some(key_code),
                native_virtual_key_code: Some(key_code),
                modifiers,
            },
            Some(session_id),
        )
        .await?;

    Ok(())
}

pub async fn scroll(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: Option<&str>,
    delta_x: f64,
    delta_y: f64,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    if let Some(sel) = selector_or_ref {
        let (object_id, effective_session_id) =
            resolve_element_object_id(client, session_id, ref_map, sel, iframe_sessions).await?;
        let js = "function(dx, dy) { this.scrollBy(dx, dy); }".to_string();
        client
            .send_command_typed::<_, Value>(
                "Runtime.callFunctionOn",
                &CallFunctionOnParams {
                    function_declaration: js,
                    object_id: Some(object_id),
                    arguments: Some(vec![
                        CallArgument {
                            value: Some(serde_json::json!(delta_x)),
                            object_id: None,
                        },
                        CallArgument {
                            value: Some(serde_json::json!(delta_y)),
                            object_id: None,
                        },
                    ]),
                    return_by_value: Some(true),
                    await_promise: Some(false),
                },
                Some(&effective_session_id),
            )
            .await?;
    } else {
        let js = format!("window.scrollBy({}, {})", delta_x, delta_y);
        client
            .send_command_typed::<_, Value>(
                "Runtime.evaluate",
                &EvaluateParams {
                    expression: js,
                    return_by_value: Some(true),
                    await_promise: Some(false),
                },
                Some(session_id),
            )
            .await?;
    }
    Ok(())
}

pub async fn select_option(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    values: &[String],
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;

    // Matching nothing must be an error, not a silent success: an agent that
    // selects a misspelled option otherwise sees "Done", and only discovers
    // the page state is wrong after more commands. List what was available.
    let js = r#"function(vals) {
            const normalize = (value) => String(value ?? '')
                .replace(/[\u200B\u200C\u200D\u2060\uFEFF]/g, '')
                .replace(/\s+/g, ' ')
                .trim();
            const options = Array.from(this.options);
            const wanted = new Set();
            for (const value of vals) {
                let matches = options.filter((opt) =>
                    value === opt.value ||
                    value === opt.label.trim() ||
                    value === opt.textContent.trim()
                );
                if (matches.length === 0) {
                    const normalizedValue = normalize(value);
                    matches = options.filter((opt) =>
                        normalize(opt.label) === normalizedValue
                    );
                    if (matches.length > 1) {
                        return { error: 'Multiple options matched ' + JSON.stringify(value) + ' after whitespace normalization' };
                    }
                }
                if (matches.length === 0) {
                    const available = options.map(o => o.value + ' ("' + normalize(o.label) + '")').join(', ');
                    return { error: 'No option matched ' + JSON.stringify(vals) + '. Available options: ' + available };
                }
                for (const opt of matches) wanted.add(opt);
            }
            if (wanted.size === 0) {
                const available = options.map(o => o.value + ' ("' + normalize(o.label) + '")').join(', ');
                return { error: 'No option matched ' + JSON.stringify(vals) + '. Available options: ' + available };
            }
            for (const opt of options) opt.selected = wanted.has(opt);
            this.dispatchEvent(new Event('change', { bubbles: true }));
            return { matched: wanted.size };
        }"#
    .to_string();

    let result = client
        .send_command_typed::<_, Value>(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration: js,
                object_id: Some(object_id),
                arguments: Some(vec![CallArgument {
                    value: Some(serde_json::json!(values)),
                    object_id: None,
                }]),
                return_by_value: Some(true),
                await_promise: Some(false),
            },
            Some(&effective_session_id),
        )
        .await?;

    if let Some(error) = result
        .get("result")
        .and_then(|r| r.get("value"))
        .and_then(|v| v.get("error"))
        .and_then(|e| e.as_str())
    {
        return Err(error.to_string());
    }

    Ok(())
}

pub async fn check(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<Option<(f64, f64)>, String> {
    let mut position = None;
    let is_checked = super::element::is_element_checked(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;
    if !is_checked {
        let result = click(
            client,
            session_id,
            ref_map,
            selector_or_ref,
            "left",
            1,
            iframe_sessions,
        )
        .await?;
        position = Some(result.position);

        // Verify the click changed the state (Playwright parity: _setChecked re-checks).
        // If the coordinate-based click missed (e.g. hidden input, overlay), retry
        // with a JS .click() on the element and its associated input.
        if !super::element::is_element_checked(
            client,
            session_id,
            ref_map,
            selector_or_ref,
            iframe_sessions,
        )
        .await?
        {
            js_click_checkbox(
                client,
                session_id,
                ref_map,
                selector_or_ref,
                iframe_sessions,
            )
            .await?;
        }
    }
    Ok(position)
}

pub async fn uncheck(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<Option<(f64, f64)>, String> {
    let mut position = None;
    let is_checked = super::element::is_element_checked(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;
    if is_checked {
        let result = click(
            client,
            session_id,
            ref_map,
            selector_or_ref,
            "left",
            1,
            iframe_sessions,
        )
        .await?;
        position = Some(result.position);

        // Same verify-and-retry as check().
        if super::element::is_element_checked(
            client,
            session_id,
            ref_map,
            selector_or_ref,
            iframe_sessions,
        )
        .await?
        {
            js_click_checkbox(
                client,
                session_id,
                ref_map,
                selector_or_ref,
                iframe_sessions,
            )
            .await?;
        }
    }
    Ok(position)
}

/// Fallback for when the coordinate-based CDP click did not toggle the
/// checkbox/radio state. This mirrors how Playwright dispatches clicks
/// through the DOM rather than via raw Input.dispatchMouseEvent coordinates.
///
/// Uses the same follow-label resolution as `is_element_checked`:
/// 1. If the element is a native input → `.click()` it directly.
/// 2. If the element is inside a `<label>` → `.click()` the label's `.control`.
/// 3. If the element has a nested `<input>` → `.click()` that input.
/// 4. Otherwise → `.click()` the element itself (handles ARIA role controls).
async fn js_click_checkbox(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;

    let js = r#"function() {
            var el = this;
            var tag = el.tagName && el.tagName.toUpperCase();
            // 1. Native input — click it directly
            if (tag === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio')) {
                el.click();
                return;
            }
            // 2. Follow label → control association
            var label = tag === 'LABEL' ? el : (el.closest && el.closest('label'));
            if (label && label.tagName && label.tagName.toUpperCase() === 'LABEL' && label.control) {
                label.control.click();
                return;
            }
            // 3. Nested native input
            var input = el.querySelector && el.querySelector('input[type="checkbox"], input[type="radio"]');
            if (input) {
                input.click();
                return;
            }
            // 4. ARIA role control — click the element itself
            el.click();
        }"#;

    client
        .send_command_typed::<_, Value>(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration: js.to_string(),
                object_id: Some(object_id),
                arguments: None,
                return_by_value: Some(true),
                await_promise: Some(false),
            },
            Some(&effective_session_id),
        )
        .await?;

    Ok(())
}

pub async fn focus(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;

    client
        .send_command_typed::<_, Value>(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration: "function() { this.focus(); }".to_string(),
                object_id: Some(object_id),
                arguments: None,
                return_by_value: Some(true),
                await_promise: Some(false),
            },
            Some(&effective_session_id),
        )
        .await?;

    Ok(())
}

pub async fn clear(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;

    client
        .send_command_typed::<_, Value>(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration: r#"function() {
                    this.focus();
                    this.value = '';
                    this.dispatchEvent(new Event('input', { bubbles: true }));
                    this.dispatchEvent(new Event('change', { bubbles: true }));
                }"#
                .to_string(),
                object_id: Some(object_id),
                arguments: None,
                return_by_value: Some(true),
                await_promise: Some(false),
            },
            Some(&effective_session_id),
        )
        .await?;

    Ok(())
}

pub async fn select_all(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;

    client
        .send_command_typed::<_, Value>(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration: r#"function() {
                    this.focus();
                    if (typeof this.select === 'function') {
                        this.select();
                    } else {
                        const range = document.createRange();
                        range.selectNodeContents(this);
                        const sel = window.getSelection();
                        sel.removeAllRanges();
                        sel.addRange(range);
                    }
                }"#
                .to_string(),
                object_id: Some(object_id),
                arguments: None,
                return_by_value: Some(true),
                await_promise: Some(false),
            },
            Some(&effective_session_id),
        )
        .await?;

    Ok(())
}

pub async fn scroll_into_view(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;

    client
        .send_command_typed::<_, Value>(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration:
                    "function() { this.scrollIntoView({ block: 'center', inline: 'center' }); }"
                        .to_string(),
                object_id: Some(object_id),
                arguments: None,
                return_by_value: Some(true),
                await_promise: Some(false),
            },
            Some(&effective_session_id),
        )
        .await?;

    Ok(())
}

pub async fn dispatch_event(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    event_type: &str,
    event_init: Option<&Value>,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;

    let init_json = event_init
        .map(|v| serde_json::to_string(v).unwrap_or("{}".to_string()))
        .unwrap_or_else(|| "{ bubbles: true }".to_string());

    let js = format!(
        "function() {{ this.dispatchEvent(new Event({}, {})); }}",
        serde_json::to_string(event_type).unwrap_or_default(),
        init_json
    );

    client
        .send_command_typed::<_, Value>(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration: js,
                object_id: Some(object_id),
                arguments: None,
                return_by_value: Some(true),
                await_promise: Some(false),
            },
            Some(&effective_session_id),
        )
        .await?;

    Ok(())
}

pub async fn highlight(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (object_id, effective_session_id) = resolve_element_object_id(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;

    client
        .send_command_typed::<_, Value>(
            "Runtime.callFunctionOn",
            &CallFunctionOnParams {
                function_declaration: r#"function() {
                    this.style.outline = '2px solid red';
                    this.style.outlineOffset = '2px';
                    const el = this;
                    setTimeout(() => {
                        el.style.outline = '';
                        el.style.outlineOffset = '';
                    }, 3000);
                }"#
                .to_string(),
                object_id: Some(object_id),
                arguments: None,
                return_by_value: Some(true),
                await_promise: Some(false),
            },
            Some(&effective_session_id),
        )
        .await?;

    Ok(())
}

pub async fn tap_touch(
    client: &CdpClient,
    session_id: &str,
    ref_map: &RefMap,
    selector_or_ref: &str,
    iframe_sessions: &HashMap<String, String>,
) -> Result<(), String> {
    let (x, y, effective_session_id) = resolve_element_center(
        client,
        session_id,
        ref_map,
        selector_or_ref,
        iframe_sessions,
    )
    .await?;

    client
        .send_command(
            "Input.dispatchTouchEvent",
            Some(serde_json::json!({
                "type": "touchStart",
                "touchPoints": [{ "x": x, "y": y }],
            })),
            Some(&effective_session_id),
        )
        .await?;

    client
        .send_command(
            "Input.dispatchTouchEvent",
            Some(serde_json::json!({
                "type": "touchEnd",
                "touchPoints": [],
            })),
            Some(&effective_session_id),
        )
        .await?;

    Ok(())
}

/// Dispatches one mouse event and waits for the browser to ack it, but
/// returns Ok(true) if a JavaScript dialog opens first. A synchronous dialog
/// (confirm/prompt/alert in the event handler) blocks the renderer's main
/// thread, so the input ack cannot arrive until the dialog is resolved;
/// without this the command hangs until the client read timeout and the agent
/// never sees the pending-dialog warning.
async fn dispatch_mouse_or_dialog(
    client: &CdpClient,
    session_id: &str,
    accept_sessions: &[&str],
    params: &DispatchMouseEventParams,
) -> Result<bool, String> {
    use tokio::sync::broadcast::error::RecvError;

    // Subscribe before sending so the dialog event cannot slip past us.
    let mut events = client.subscribe();
    let send =
        client.send_command_typed::<_, Value>("Input.dispatchMouseEvent", params, Some(session_id));
    tokio::pin!(send);
    loop {
        tokio::select! {
            res = &mut send => {
                res?;
                return Ok(false);
            }
            event = events.recv() => {
                match event {
                    Ok(e) if e.method == "Page.javascriptDialogOpening" => {
                        // Only a dialog on this click's frame/page session
                        // aborts it; a background-tab dialog must not. A
                        // session-less event has no flat session and is
                        // treated as the top-level page (i.e. ours).
                        let ours = match e.session_id.as_deref() {
                            Some(sid) => accept_sessions.contains(&sid),
                            None => true,
                        };
                        if ours {
                            return Ok(true);
                        }
                        continue;
                    }
                    Ok(_) => continue,
                    Err(RecvError::Lagged(_)) => continue,
                    Err(RecvError::Closed) => {
                        (&mut send).await?;
                        return Ok(false);
                    }
                }
            }
        }
    }
}

async fn dispatch_click(
    client: &CdpClient,
    session_id: &str,
    accept_sessions: &[&str],
    x: f64,
    y: f64,
    button: &str,
    click_count: i32,
) -> Result<ClickResult, String> {
    // Move
    if dispatch_mouse_or_dialog(
        client,
        session_id,
        accept_sessions,
        &DispatchMouseEventParams {
            event_type: "mouseMoved".to_string(),
            x,
            y,
            button: None,
            buttons: None,
            click_count: None,
            delta_x: None,
            delta_y: None,
            modifiers: None,
        },
    )
    .await?
    {
        // No button was pressed yet, nothing to release.
        return Ok(ClickResult {
            position: (x, y),
            dialog_opened: true,
            pending_release: None,
            x,
            y,
            button_pressed: false,
        });
    }

    let button_value = match button {
        "right" => 2,
        "middle" => 4,
        _ => 1,
    };

    // Press
    if dispatch_mouse_or_dialog(
        client,
        session_id,
        accept_sessions,
        &DispatchMouseEventParams {
            event_type: "mousePressed".to_string(),
            x,
            y,
            button: Some(button.to_string()),
            buttons: Some(button_value),
            click_count: Some(click_count),
            delta_x: None,
            delta_y: None,
            modifiers: None,
        },
    )
    .await?
    {
        // Dialog opened from the mousedown handler: the button is held and the
        // release will never arrive on its own. Hand the caller what it needs
        // to release once the dialog is resolved.
        return Ok(ClickResult {
            position: (x, y),
            dialog_opened: true,
            pending_release: Some(PendingRelease {
                session_id: session_id.to_string(),
                x,
                y,
                button: button.to_string(),
            }),
            x,
            y,
            button_pressed: true,
        });
    }

    // Release. A dialog here fired from the click/mouseup handler, which runs
    // after the button is already up, so there is nothing left to release.
    let dialog_opened = dispatch_mouse_or_dialog(
        client,
        session_id,
        accept_sessions,
        &DispatchMouseEventParams {
            event_type: "mouseReleased".to_string(),
            x,
            y,
            button: Some(button.to_string()),
            buttons: Some(0),
            click_count: Some(click_count),
            delta_x: None,
            delta_y: None,
            modifiers: None,
        },
    )
    .await?;
    Ok(ClickResult {
        position: (x, y),
        dialog_opened,
        pending_release: None,
        x,
        y,
        button_pressed: true,
    })
}

/// Best-effort mouseReleased to clear a button left logically down when a
/// dialog opened mid-click. Called after the dialog is resolved.
pub async fn dispatch_pending_release(
    client: &CdpClient,
    release: &PendingRelease,
) -> Result<(), String> {
    client
        .send_command_typed::<_, Value>(
            "Input.dispatchMouseEvent",
            &DispatchMouseEventParams {
                event_type: "mouseReleased".to_string(),
                x: release.x,
                y: release.y,
                button: Some(release.button.clone()),
                buttons: Some(0),
                click_count: Some(1),
                delta_x: None,
                delta_y: None,
                modifiers: None,
            },
            Some(&release.session_id),
        )
        .await?;
    Ok(())
}

fn char_to_key_info(ch: char) -> (String, String, i32) {
    match ch {
        '\n' | '\r' => ("Enter".to_string(), "Enter".to_string(), 13),
        '\t' => ("Tab".to_string(), "Tab".to_string(), 9),
        ' ' => (" ".to_string(), "Space".to_string(), 32),
        _ => {
            let key = ch.to_string();
            if ch.is_ascii_alphabetic() {
                // For letters the Windows VK code equals the uppercase ASCII value.
                let upper = ch.to_ascii_uppercase();
                let code = format!("Key{}", upper);
                let key_code = upper as i32;
                (key, code, key_code)
            } else if ch.is_ascii_digit() {
                let code = format!("Digit{}", ch);
                let key_code = ch as i32;
                (key, code, key_code)
            } else {
                let (code, key_code) = punctuation_key_info(ch);
                (key, code.to_string(), key_code)
            }
        }
    }
}

/// Return the DOM `KeyboardEvent.code` value and Windows virtual-key code for
/// a punctuation / symbol character assuming a US keyboard layout.
///
/// The Windows virtual-key codes (VK_OEM_*) differ from ASCII values for
/// punctuation.  Using the raw ASCII code would misidentify characters – e.g.
/// '.' (ASCII 46) collides with VK_DELETE (0x2E = 46), causing the period to
/// be swallowed.
fn punctuation_key_info(ch: char) -> (&'static str, i32) {
    match ch {
        // VK_OEM_1 (0xBA = 186) — ";:" key on US layout
        ';' | ':' => ("Semicolon", 186),
        // VK_OEM_PLUS (0xBB = 187) — "=+" key
        '=' | '+' => ("Equal", 187),
        // VK_OEM_COMMA (0xBC = 188) — ",<" key
        ',' | '<' => ("Comma", 188),
        // VK_OEM_MINUS (0xBD = 189) — "-_" key
        '-' | '_' => ("Minus", 189),
        // VK_OEM_PERIOD (0xBE = 190) — ".>" key
        '.' | '>' => ("Period", 190),
        // VK_OEM_2 (0xBF = 191) — "/?" key
        '/' | '?' => ("Slash", 191),
        // VK_OEM_3 (0xC0 = 192) — "`~" key
        '`' | '~' => ("Backquote", 192),
        // VK_OEM_4 (0xDB = 219) — "[{" key
        '[' | '{' => ("BracketLeft", 219),
        // VK_OEM_5 (0xDC = 220) — "\\|" key
        '\\' | '|' => ("Backslash", 220),
        // VK_OEM_6 (0xDD = 221) — "]}" key
        ']' | '}' => ("BracketRight", 221),
        // VK_OEM_7 (0xDE = 222) — "'\""" key
        '\'' | '"' => ("Quote", 222),
        _ => ("", 0),
    }
}

/// Return the `text` value that CDP `Input.dispatchKeyEvent` needs on the
/// `keyDown` event so that Chrome performs the default action for the key.
/// For example Enter needs `"\r"` to actually submit a form, and Tab needs
/// `"\t"` to move focus.  Non-printable / navigation keys return `None`.
fn key_text(key_name: &str) -> Option<String> {
    match key_name {
        "Enter" => Some("\r".to_string()),
        "Tab" => Some("\t".to_string()),
        " " => Some(" ".to_string()),
        _ => {
            // Single printable characters carry themselves as text.
            if key_name.len() == 1 {
                Some(key_name.to_string())
            } else {
                None
            }
        }
    }
}

fn named_key_info(key: &str) -> (String, String, i32) {
    match key.to_lowercase().as_str() {
        "enter" | "return" => ("Enter".to_string(), "Enter".to_string(), 13),
        "tab" => ("Tab".to_string(), "Tab".to_string(), 9),
        "escape" | "esc" => ("Escape".to_string(), "Escape".to_string(), 27),
        "backspace" => ("Backspace".to_string(), "Backspace".to_string(), 8),
        "delete" => ("Delete".to_string(), "Delete".to_string(), 46),
        "arrowup" | "up" => ("ArrowUp".to_string(), "ArrowUp".to_string(), 38),
        "arrowdown" | "down" => ("ArrowDown".to_string(), "ArrowDown".to_string(), 40),
        "arrowleft" | "left" => ("ArrowLeft".to_string(), "ArrowLeft".to_string(), 37),
        "arrowright" | "right" => ("ArrowRight".to_string(), "ArrowRight".to_string(), 39),
        "home" => ("Home".to_string(), "Home".to_string(), 36),
        "end" => ("End".to_string(), "End".to_string(), 35),
        "pageup" => ("PageUp".to_string(), "PageUp".to_string(), 33),
        "pagedown" => ("PageDown".to_string(), "PageDown".to_string(), 34),
        "space" | " " => (" ".to_string(), "Space".to_string(), 32),
        _ => {
            if key.len() == 1 {
                let ch = key.chars().next().unwrap();
                char_to_key_info(ch)
            } else {
                (key.to_string(), key.to_string(), 0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `char_to_key_info` returns the correct (key, code,
    /// windowsVirtualKeyCode) triple for every character in Playwright's
    /// USKeyboardLayout.  The expected values below are taken verbatim from
    /// playwright-core/lib/server/usKeyboardLayout.js so that any drift from
    /// Playwright's behaviour is caught immediately.
    #[test]
    fn test_char_to_key_info_matches_playwright_layout() {
        // (character, expected_code, expected_vk_code)
        let cases: &[(char, &str, i32)] = &[
            // Letters – VK code must equal the uppercase ASCII value.
            ('a', "KeyA", 65),
            ('z', "KeyZ", 90),
            ('A', "KeyA", 65),
            // Digits
            ('0', "Digit0", 48),
            ('9', "Digit9", 57),
            // Punctuation – these are the values from Playwright's layout.
            // The bug that prompted this test sent '.' as VK 46 (= VK_DELETE).
            ('.', "Period", 190),
            (',', "Comma", 188),
            ('/', "Slash", 191),
            (';', "Semicolon", 186),
            ('\'', "Quote", 222),
            ('[', "BracketLeft", 219),
            (']', "BracketRight", 221),
            ('\\', "Backslash", 220),
            ('`', "Backquote", 192),
            ('-', "Minus", 189),
            ('=', "Equal", 187),
            // Shifted variants produced by the same physical keys.
            ('>', "Period", 190),
            ('<', "Comma", 188),
            ('?', "Slash", 191),
            (':', "Semicolon", 186),
            ('"', "Quote", 222),
            ('{', "BracketLeft", 219),
            ('}', "BracketRight", 221),
            ('|', "Backslash", 220),
            ('~', "Backquote", 192),
            ('_', "Minus", 189),
            ('+', "Equal", 187),
            // Whitespace / control
            (' ', "Space", 32),
            ('\n', "Enter", 13),
            ('\t', "Tab", 9),
        ];

        for &(ch, expected_code, expected_vk) in cases {
            let (key, code, vk) = char_to_key_info(ch);
            assert_eq!(
                code, expected_code,
                "char {:?}: expected code {:?}, got {:?}",
                ch, expected_code, code
            );
            assert_eq!(
                vk, expected_vk,
                "char {:?}: expected VK {}, got {} (ASCII would be {})",
                ch, expected_vk, vk, ch as i32
            );
            // key should be the character itself (except control chars).
            if !ch.is_control() {
                assert_eq!(key, ch.to_string(), "char {:?}: key mismatch", ch);
            }
        }
    }

    /// Regression test: period must NEVER map to VK 46 (VK_DELETE).
    #[test]
    fn test_period_is_not_vk_delete() {
        let (_, _, vk) = char_to_key_info('.');
        assert_ne!(
            vk, 46,
            "Period must not use VK code 46 (VK_DELETE); expected 190 (VK_OEM_PERIOD)"
        );
        assert_eq!(vk, 190);
    }

    /// Characters outside the US keyboard layout should return (key, "", 0)
    /// so that `type_text` falls back to `Input.insertText`.
    #[test]
    fn test_unmapped_chars_return_zero_keycode() {
        for ch in ['@', '#', '$', '%', '^', '&', '*', '(', ')', '€', '£', '你'] {
            let (key, code, vk) = char_to_key_info(ch);
            assert_eq!(
                code, "",
                "char {:?}: unmapped char should have empty code, got {:?}",
                ch, code
            );
            assert_eq!(
                vk, 0,
                "char {:?}: unmapped char should have VK 0, got {}",
                ch, vk
            );
            assert_eq!(key, ch.to_string());
        }
    }

    /// Expected outcomes come from Playwright's `fill` (injectedScript.ts):
    /// which input types get their value set, which are typed into, and the
    /// messages for everything else.
    #[test]
    fn test_text_entry_follows_playwright_fill_contract() {
        let set = |t: &str| Ok(TextEntry::SetValue(t.to_string()));
        let not_fillable = |t: &str| Err(format!("Input '@e1' of type \"{t}\" cannot be filled"));
        let not_text = || {
            Err("Element '@e1' is not an <input>, <textarea> or [contenteditable] element".into())
        };
        let cases: &[(&str, &str, bool, Result<TextEntry, String>)] = &[
            ("input", "color", false, set("color")),
            ("input", "date", false, set("date")),
            ("input", "time", false, set("time")),
            ("input", "datetime-local", false, set("datetime-local")),
            ("input", "month", false, set("month")),
            ("input", "range", false, set("range")),
            ("input", "week", false, set("week")),
            ("input", "", false, Ok(TextEntry::Type)),
            ("input", "email", false, Ok(TextEntry::Type)),
            ("input", "number", false, Ok(TextEntry::Type)),
            ("input", "password", false, Ok(TextEntry::Type)),
            ("input", "search", false, Ok(TextEntry::Type)),
            ("input", "tel", false, Ok(TextEntry::Type)),
            ("input", "text", false, Ok(TextEntry::Type)),
            ("input", "url", false, Ok(TextEntry::Type)),
            ("input", "checkbox", false, not_fillable("checkbox")),
            ("input", "radio", false, not_fillable("radio")),
            ("input", "file", false, not_fillable("file")),
            ("input", "submit", false, not_fillable("submit")),
            ("input", "hidden", false, not_fillable("hidden")),
            ("textarea", "", false, Ok(TextEntry::Type)),
            ("div", "", true, Ok(TextEntry::Type)),
            ("select", "", false, not_text()),
            ("button", "", false, not_text()),
            ("div", "", false, not_text()),
        ];
        for (tag, input_type, editable, expected) in cases {
            assert_eq!(
                &text_entry("@e1", tag, input_type, *editable, false),
                expected,
                "<{tag}> type={input_type:?} editable={editable}"
            );
        }
        // Not from Playwright, which refuses frames: a frame the page can't
        // read takes text wherever focus is inside it, as agent-browser has
        // always typed there. One it can read is refused like any other
        // element once nothing focused inside it takes text.
        let frames: &[(&str, bool, Result<TextEntry, String>)] = &[
            ("iframe", true, Ok(TextEntry::Type)),
            ("frame", true, Ok(TextEntry::Type)),
            ("iframe", false, not_text()),
        ];
        for (tag, unreadable_frame, expected) in frames {
            assert_eq!(
                &text_entry("@e1", tag, "", false, *unreadable_frame),
                expected,
                "<{tag}> unreadable_frame={unreadable_frame}"
            );
        }
    }

    #[test]
    fn test_key_text_returns_correct_text_for_special_keys() {
        assert_eq!(key_text("Enter"), Some("\r".to_string()));
        assert_eq!(key_text("Tab"), Some("\t".to_string()));
        assert_eq!(key_text(" "), Some(" ".to_string()));
        // Single printable characters carry themselves.
        assert_eq!(key_text("a"), Some("a".to_string()));
        assert_eq!(key_text("Z"), Some("Z".to_string()));
        // Non-printable named keys return None.
        assert_eq!(key_text("Escape"), None);
        assert_eq!(key_text("ArrowUp"), None);
        assert_eq!(key_text("Backspace"), None);
        assert_eq!(key_text("Delete"), None);
    }
}

//! End-to-end tests for the native daemon.
//!
//! These tests launch a real Chrome instance and exercise the full command
//! pipeline. They require Chrome to be installed and are marked `#[ignore]`
//! so they don't run during normal `cargo test`.
//!
//! Run serially to avoid Chrome instance contention:
//!   cargo test e2e -- --ignored --test-threads=1

use base64::{engine::general_purpose::STANDARD, Engine};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::test_utils::EnvGuard;

use super::actions::{
    close_current_browser, execute_command, maybe_autosave_restore_state, DaemonState,
};

fn assert_success(resp: &Value) {
    assert_eq!(
        resp.get("success").and_then(|v| v.as_bool()),
        Some(true),
        "Expected success but got: {}",
        serde_json::to_string_pretty(resp).unwrap_or_default()
    );
}

fn get_data(resp: &Value) -> &Value {
    resp.get("data").expect("Missing 'data' in response")
}

async fn select_values(
    state: &mut DaemonState,
    id: &str,
    selector: &str,
    values: &[&str],
) -> Value {
    execute_command(
        &json!({ "id": id, "action": "select", "selector": selector, "values": values }),
        state,
    )
    .await
}

async fn assert_evaluate(state: &mut DaemonState, id: &str, script: &str, expected: Value) {
    let resp = execute_command(
        &json!({ "id": id, "action": "evaluate", "script": script }),
        state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], expected);
}

fn assert_error_code(resp: &Value, code: &str) {
    assert_eq!(
        resp.get("success").and_then(Value::as_bool),
        Some(false),
        "Expected failure but got: {}",
        serde_json::to_string_pretty(resp).unwrap_or_default()
    );
    assert_eq!(resp.get("code").and_then(Value::as_str), Some(code));
}

fn native_test_fixture_html(name: &str) -> &'static str {
    match name {
        "drag_probe" => include_str!("test_fixtures/drag_probe.html"),
        "html5_drag_probe" => include_str!("test_fixtures/html5_drag_probe.html"),
        "pointer_capture_probe" => include_str!("test_fixtures/pointer_capture_probe.html"),
        "snapshot_diff_probe" => include_str!("test_fixtures/snapshot_diff_probe.html"),
        "upload_probe" => include_str!("test_fixtures/upload_probe.html"),
        "webmcp_delayed_probe" => include_str!("test_fixtures/webmcp_delayed_probe.html"),
        "webmcp_frame_probe" => include_str!("test_fixtures/webmcp_frame_probe.html"),
        "webmcp_probe" => include_str!("test_fixtures/webmcp_probe.html"),
        "webmcp_context_probe" => include_str!("test_fixtures/webmcp_context_probe.html"),
        _ => panic!("Unknown native test fixture: {}", name),
    }
}

#[tokio::test]
#[ignore]
async fn e2e_webmcp_discovery_invocation_and_cancellation() {
    let (fixture_url, fixture_server) = start_webmcp_server().await;
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": fixture_url
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["webmcp"]["experimental"], true);
    assert_eq!(get_data(&resp)["webmcp"]["available"], true);
    assert!(
        get_data(&resp)["webmcp"]["toolCount"]
            .as_u64()
            .is_some_and(|count| count >= 4),
        "navigation did not advertise the fixture's WebMCP tools: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );
    let child_ready = tokio::time::timeout(tokio::time::Duration::from_secs(5), async {
        loop {
            let resp = execute_command(
                &json!({
                    "id": "2b",
                    "action": "evaluate",
                    "script": "document.getElementById('tool-frame')?.contentDocument?.body?.dataset?.webmcpReady === 'true'"
                }),
                &mut state,
            )
            .await;
            if get_data(&resp)["result"] == true {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(25)).await;
        }
    })
    .await;
    assert!(
        child_ready.is_ok(),
        "child WebMCP fixture did not become ready"
    );
    state.drain_cdp_events_background().await.unwrap();

    let resp = execute_command(&json!({ "id": "3", "action": "webmcp_list" }), &mut state).await;
    assert_success(&resp);
    let tools = get_data(&resp)["tools"].as_array().unwrap();
    let tool = tools
        .iter()
        .find(|tool| tool["name"] == "set_message")
        .expect("set_message should be discovered");
    assert!(tool["frameId"].as_str().is_some_and(|id| !id.is_empty()));
    assert!(tool["origin"]
        .as_str()
        .is_some_and(|origin| origin.starts_with("http://127.0.0.1:")));
    assert_eq!(tool["inputSchema"]["type"], "object");
    let duplicate_tools = tools
        .iter()
        .filter(|tool| tool["name"] == "duplicate_tool")
        .collect::<Vec<_>>();
    assert_eq!(
        duplicate_tools.len(),
        2,
        "unexpected tools: {}",
        serde_json::to_string_pretty(tools).unwrap_or_default()
    );
    let main_frame_id = tool["frameId"].as_str().unwrap();
    let child_frame_id = duplicate_tools
        .iter()
        .find_map(|tool| {
            let frame_id = tool["frameId"].as_str()?;
            (frame_id != main_frame_id).then_some(frame_id)
        })
        .unwrap()
        .to_string();

    let resp = execute_command(
        &json!({
            "id": "3b",
            "action": "webmcp_invoke",
            "tool": "duplicate_tool",
            "params": {}
        }),
        &mut state,
    )
    .await;
    assert_error_code(&resp, "webmcp_ambiguous_tool");

    let resp = execute_command(
        &json!({
            "id": "3c",
            "action": "webmcp_invoke",
            "tool": "duplicate_tool",
            "frameId": child_frame_id,
            "params": {},
            "timeout": 5000
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["output"]["scope"], "frame");

    let resp = execute_command(
        &json!({
            "id": "4",
            "action": "webmcp_invoke",
            "tool": "set_message",
            "params": { "message": "WebMCP works" },
            "timeout": 5000
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["status"], "completed");
    assert_eq!(get_data(&resp)["output"]["message"], "WebMCP works");

    let resp = execute_command(
        &json!({
            "id": "4b",
            "action": "webmcp_invoke",
            "tool": "fail_tool",
            "params": {},
            "timeout": 5000
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["status"], "failed");
    assert!(get_data(&resp)["error"].is_string());

    let resp = execute_command(
        &json!({
            "id": "5",
            "action": "evaluate",
            "script": "document.getElementById('result').textContent"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "WebMCP works");

    let resp = execute_command(
        &json!({
            "id": "6",
            "action": "webmcp_invoke",
            "tool": "wait_for_cancel",
            "params": {},
            "detach": true
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let invocation_id = get_data(&resp)["invocationId"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(get_data(&resp)["status"], "pending");
    let pending = execute_command(
        &json!({
            "id": "6b",
            "action": "webmcp_result",
            "invocationId": invocation_id,
            "timeout": 10
        }),
        &mut state,
    )
    .await;
    assert_success(&pending);
    assert_eq!(get_data(&pending)["status"], "timed_out");

    let resp = execute_command(
        &json!({
            "id": "6c",
            "action": "webmcp_invoke",
            "tool": "wait_for_cancel",
            "params": {},
            "detach": true
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let invocation_id = get_data(&resp)["invocationId"]
        .as_str()
        .unwrap()
        .to_string();
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    state.drain_cdp_events_background().await.unwrap();
    assert_eq!(
        state.webmcp.invocations[&invocation_id].status,
        "pending",
        "long-running fixture terminated before cancellation: {}",
        state.webmcp.invocations[&invocation_id].to_json()
    );

    let resp = execute_command(
        &json!({
            "id": "7",
            "action": "webmcp_cancel",
            "invocationId": invocation_id,
            "timeout": 5000
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["status"], "canceled");

    let resp = execute_command(
        &json!({
            "id": "8",
            "action": "webmcp_invoke",
            "tool": "wait_for_cancel",
            "params": {},
            "timeout": 25
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["status"], "timed_out");

    let resp = execute_command(
        &json!({
            "id": "9",
            "action": "webmcp_invoke",
            "tool": "missing_tool",
            "params": {}
        }),
        &mut state,
    )
    .await;
    assert_error_code(&resp, "webmcp_tool_not_found");

    let resp = execute_command(
        &json!({
            "id": "10",
            "action": "webmcp_invoke",
            "tool": "set_message",
            "params": ["not", "an", "object"]
        }),
        &mut state,
    )
    .await;
    assert_error_code(&resp, "webmcp_invalid_input");

    let resp = execute_command(
        &json!({
            "id": "11",
            "action": "webmcp_invoke",
            "tool": "wait_for_cancel",
            "params": {},
            "detach": true
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let stale_id = get_data(&resp)["invocationId"]
        .as_str()
        .unwrap()
        .to_string();
    let resp = execute_command(
        &json!({
            "id": "12",
            "action": "navigate",
            "url": "about:blank"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({
            "id": "13",
            "action": "webmcp_result",
            "invocationId": stale_id,
            "timeout": 100
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["status"], "failed");
    assert!(get_data(&resp)["error"]
        .as_str()
        .is_some_and(|error| error.starts_with("webmcp_context_changed:")));

    let resp = execute_command(
        &json!({
            "id": "14",
            "action": "navigate",
            "url": fixture_url
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(&json!({ "id": "15", "action": "webmcp_list" }), &mut state).await;
    assert_success(&resp);
    let frame_id = get_data(&resp)["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "frame_wait")
        .and_then(|tool| tool["frameId"].as_str())
        .unwrap()
        .to_string();
    let resp = execute_command(
        &json!({
            "id": "16",
            "action": "webmcp_invoke",
            "tool": "frame_wait",
            "frameId": frame_id,
            "params": {},
            "detach": true
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let frame_invocation_id = get_data(&resp)["invocationId"]
        .as_str()
        .unwrap()
        .to_string();
    let resp = execute_command(
        &json!({
            "id": "17",
            "action": "evaluate",
            "script": "document.getElementById('tool-frame').remove()"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    let resp = execute_command(
        &json!({
            "id": "18",
            "action": "webmcp_result",
            "invocationId": frame_invocation_id,
            "timeout": 100
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["status"], "failed");
    assert!(get_data(&resp)["error"]
        .as_str()
        .is_some_and(|error| error.starts_with("webmcp_context_changed:")));

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    assert!(state.webmcp.invocations.is_empty());
    fixture_server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_webmcp_delayed_registration_appears_on_next_action() {
    let (fixture_url, fixture_server) = start_webmcp_server().await;
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": format!("{fixture_url}/delayed.html")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let already_observed = get_data(&resp).get("webmcp").is_some();
    let resp = execute_command(
        &json!({"id": "wait", "action": "wait", "timeout": 150}),
        &mut state,
    )
    .await;
    assert_success(&resp);
    if !already_observed {
        assert_eq!(get_data(&resp)["webmcp"]["experimental"], true);
        assert_eq!(get_data(&resp)["webmcp"]["available"], true);
        assert_eq!(get_data(&resp)["webmcp"]["toolCount"], 1);
    }

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    fixture_server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_webmcp_context_follows_actions_without_explicit_discovery() {
    let (url, server) = start_webmcp_server().await;
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({"id": "launch", "action": "launch", "headless": true}),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let commands = [
        (
            json!({"action": "navigate", "url": format!("{url}/context.html")}),
            -1,
        ),
        (json!({"action": "click", "selector": "#register"}), 1),
        (json!({"action": "snapshot"}), -1),
        (json!({"action": "webmcp_list", "tool": "set_message"}), -1),
        (
            json!({"action": "fill", "selector": "#description", "value": "Updated description"}),
            1,
        ),
        (
            json!({"action": "webmcp_invoke", "tool": "set_message", "params": {"message": "Hello from discovered tool"}}),
            -1,
        ),
        (json!({"action": "gettext", "selector": "#result"}), -1),
        (
            json!({"action": "evaluate", "script": "history.pushState({}, '', '#route')"}),
            -1,
        ),
        (
            json!({"action": "tab_new", "url": format!("{url}/empty.html")}),
            0,
        ),
        (json!({"action": "tab_switch", "tabId": "t1"}), 1),
        (json!({"action": "tab_close", "tabId": "t2"}), -1),
        (json!({"action": "click", "selector": "#remove"}), 0),
        (json!({"action": "wait", "timeout": 50}), -1),
        (json!({"action": "click", "selector": "#register"}), 1),
        (
            json!({"action": "navigate", "url": format!("{url}/empty.html")}),
            0,
        ),
    ];
    for (mut cmd, count) in commands {
        cmd["id"] = json!("context");
        let resp = execute_command(&cmd, &mut state).await;
        assert_success(&resp);
        let mut context = get_data(&resp)["webmcp"].clone();
        // Chrome may deliver tool events after the action's reply. Passive
        // observation reports them on a later ordinary action, without listing
        // tools or retrying the mutation. Re-registration can also briefly
        // report removal before the replacement tool is observed.
        if count >= 0 && context["toolCount"] != count {
            context = tokio::time::timeout(std::time::Duration::from_secs(1), async {
                loop {
                    let update = execute_command(
                        &json!({"id": "context-update", "action": "wait", "timeout": 50}),
                        &mut state,
                    )
                    .await;
                    assert_success(&update);
                    let context = &get_data(&update)["webmcp"];
                    if context["toolCount"] == count {
                        break context.clone();
                    }
                }
            })
            .await
            .unwrap_or_else(|_| panic!("Missing WebMCP update after {cmd}: {resp}"));
        }
        if count == -1 {
            assert!(
                context.is_null(),
                "Unchanged or empty page must stay quiet: {cmd}: {resp}"
            );
        } else {
            assert_eq!(context["status"], "ready", "{cmd}: {resp}");
            assert_eq!(context["toolCount"], count, "{cmd}: {resp}");
            assert_eq!(context["available"], count > 0);
        }
        if count > 0 {
            let tool = &context["tools"][0];
            assert_eq!(tool["name"], "set_message");
            assert!(tool.get("inputSchema").is_none());
            assert!(tool["frameId"].as_str().is_some_and(|id| !id.is_empty()));
            assert_eq!(tool["origin"], url);
            if cmd["action"] == "fill" {
                assert_eq!(tool["description"], "Updated description");
            }
        }
        if cmd["action"] == "webmcp_list" {
            assert_eq!(get_data(&resp)["tools"].as_array().unwrap().len(), 1);
            assert_eq!(
                get_data(&resp)["tools"][0]["inputSchema"]["required"],
                json!(["message"])
            );
        }
        if cmd["action"] == "gettext" {
            assert_eq!(get_data(&resp)["text"], "Hello from discovered tool");
        }
    }
    let resp = execute_command(&json!({"id": "close", "action": "close"}), &mut state).await;
    assert_success(&resp);
    assert!(get_data(&resp).get("webmcp").is_none());
    server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_webmcp_same_document_navigation_preserves_catalog_and_invocations() {
    let (url, server) = start_webmcp_server().await;
    let mut state = DaemonState::new();
    for mut cmd in [
        json!({"action": "launch", "headless": true}),
        json!({"action": "navigate", "url": url}),
    ] {
        cmd["id"] = json!("setup");
        let resp = execute_command(&cmd, &mut state).await;
        assert_success(&resp);
    }
    let resp = execute_command(
        &json!({"id": "invoke", "action": "webmcp_invoke", "tool": "wait_for_cancel", "params": {}, "detach": true}),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let invocation_id = get_data(&resp)["invocationId"]
        .as_str()
        .unwrap()
        .to_string();
    let browser = state.browser.as_ref().unwrap();
    let session = browser.active_session_id().unwrap().to_string();
    // Stop tool events so this test cannot pass by silently rediscovering the
    // catalog. Page navigation events remain enabled for document invalidation.
    browser
        .client
        .send_command_no_params("WebMCP.disable", Some(&session))
        .await
        .unwrap();
    let resp = execute_command(&json!({"id": "settle", "action": "title"}), &mut state).await;
    assert_success(&resp);
    let tools = state.webmcp.tools(&session).unwrap();
    assert!(tools.iter().any(|tool| tool.name == "wait_for_cancel"));
    assert_eq!(state.webmcp.invocations[&invocation_id].status, "pending");

    for mut cmd in [
        json!({"action": "navigate", "url": format!("{url}/#route")}),
        json!({"action": "snapshot"}),
        json!({"action": "navigate", "url": format!("{url}/#next")}),
        json!({"action": "evaluate", "script": "history.pushState({}, '', '#history')"}),
        json!({"action": "title"}),
    ] {
        cmd["id"] = json!("same-document");
        let resp = execute_command(&cmd, &mut state).await;
        assert_success(&resp);
        assert!(get_data(&resp).get("webmcp").is_none(), "{cmd}: {resp}");
        assert_eq!(state.webmcp.tools(&session).unwrap(), tools, "{cmd}");
        assert_eq!(state.webmcp.invocations[&invocation_id].status, "pending");
    }

    let resp = execute_command(
        &json!({"id": "new-document", "action": "navigate", "url": format!("{url}/empty.html")}),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["webmcp"]["toolCount"], 0);
    assert!(state.webmcp.tools(&session).unwrap().is_empty());
    assert!(state.webmcp.invocations[&invocation_id].to_json()["error"]
        .as_str()
        .is_some_and(|error| error.starts_with("webmcp_context_changed:")));

    // New documents still populate the catalog through the existing
    // subscription and reannounce tools even when the records are identical.
    state
        .browser
        .as_ref()
        .unwrap()
        .client
        .send_command_no_params("WebMCP.enable", Some(&session))
        .await
        .unwrap();
    for _ in 0..2 {
        let resp = execute_command(
            &json!({"id": "reload", "action": "navigate", "url": url}),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert!(get_data(&resp)["webmcp"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "wait_for_cancel"));
    }
    let resp = execute_command(&json!({"id": "close", "action": "close"}), &mut state).await;
    assert_success(&resp);
    server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_webmcp_ordinary_page_stays_quiet_without_reprobing() {
    let (url, server) = start_webmcp_server().await;
    let mut state = DaemonState::new();
    for (index, mut cmd) in [
        json!({"action": "launch", "headless": true}),
        json!({"action": "navigate", "url": format!("{url}/empty.html")}),
        json!({"action": "snapshot"}),
        json!({"action": "title"}),
        json!({"action": "evaluate", "script": "document.title"}),
        json!({"action": "wait", "timeout": 1}),
    ]
    .into_iter()
    .enumerate()
    {
        cmd["id"] = json!(index.to_string());
        let resp = execute_command(&cmd, &mut state).await;
        assert_success(&resp);
        assert!(get_data(&resp).get("webmcp").is_none(), "{cmd}: {resp}");
    }
    // Disable browser-side events behind the daemon. If ordinary commands
    // re-enable discovery, the registration below would incorrectly appear.
    let browser = state.browser.as_ref().unwrap();
    let session = browser.active_session_id().unwrap().to_string();
    browser
        .client
        .send_command_no_params("WebMCP.disable", Some(&session))
        .await
        .unwrap();
    let resp = execute_command(&json!({"id": "register", "action": "evaluate", "script": "document.modelContext.registerTool({name: 'probe', description: 'probe', inputSchema: {type: 'object'}, execute: async () => ({ok: true})}); true"}), &mut state).await;
    assert_success(&resp);
    assert!(get_data(&resp).get("webmcp").is_none());
    assert_eq!(state.webmcp.observations.len(), 1);
    let resp = execute_command(&json!({"id": "snapshot", "action": "snapshot"}), &mut state).await;
    assert_success(&resp);
    assert!(get_data(&resp).get("webmcp").is_none());
    execute_command(&json!({"id": "close", "action": "close"}), &mut state).await;
    server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_webmcp_opt_out_returns_no_tools() {
    let (fixture_url, fixture_server) = start_webmcp_server().await;
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({
            "id": "1",
            "action": "launch",
            "headless": true,
            "webmcp": false
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "2", "action": "webmcp_list" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["tools"], json!([]));

    let resp = execute_command(
        &json!({ "id": "3", "action": "navigate", "url": fixture_url }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(get_data(&resp).get("webmcp").is_none());

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    fixture_server.abort();
}

fn native_test_fixture_url(name: &str) -> String {
    format!(
        "data:text/html;base64,{}",
        STANDARD.encode(native_test_fixture_html(name))
    )
}

async fn create_storage_state_with_cookie(path: &str, cookie_name: &str, cookie_value: &str) {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({
            "id": "1",
            "action": "launch",
            "headless": true,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"]
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "cookies_set",
            "name": cookie_name,
            "value": cookie_value,
            "domain": ".example.com",
            "path": "/",
            "expires": 2000000000
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "4", "action": "state_save", "path": path }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "5", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

async fn create_restore_state_with_cookie(
    restore_key: &str,
    cookie_name: &str,
    cookie_value: &str,
) {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({
            "id": "1",
            "action": "navigate",
            "url": "https://example.com",
            "restoreKey": restore_key
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "cookies_set",
            "name": cookie_name,
            "value": cookie_value,
            "domain": ".example.com",
            "path": "/",
            "expires": 2000000000
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "3", "action": "close" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["saveStatus"], "saved");
}

fn cleanup_restore_state_files(restore_key: &str) {
    let Some(sessions_dir) = dirs::home_dir().map(|home| home.join(".agent-browser/sessions"))
    else {
        return;
    };

    if let Ok(entries) = std::fs::read_dir(&sessions_dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname.starts_with(&format!("{}-", restore_key)) {
                let path = entry.path();
                let _ = std::fs::remove_file(&path);
                let _ = std::fs::remove_file(format!("{}.previous", path.to_string_lossy()));
            }
        }
    }
}

async fn send_raw_http_request(port: u64, request: &str) -> String {
    let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .expect("HTTP client should connect to stream server");
    stream
        .write_all(request.as_bytes())
        .await
        .expect("HTTP request should be written");
    stream
        .shutdown()
        .await
        .expect("HTTP client write side should shut down");

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .expect("HTTP response should be read");
    String::from_utf8(response).expect("HTTP response should be utf-8")
}

#[cfg(unix)]
async fn spawn_fake_daemon_socket(
    socket_dir: &std::path::Path,
    session_name: &str,
) -> tokio::sync::oneshot::Receiver<String> {
    use tokio::io::AsyncBufReadExt;

    let socket_path = socket_dir.join(format!("{session_name}.sock"));
    let _ = std::fs::remove_file(&socket_path);
    let listener =
        tokio::net::UnixListener::bind(&socket_path).expect("fake daemon socket should bind");
    let (tx, rx) = tokio::sync::oneshot::channel();

    tokio::spawn(async move {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let mut reader = tokio::io::BufReader::new(stream);
        let mut command = String::new();
        if reader.read_line(&mut command).await.is_err() {
            return;
        }

        let mut stream = reader.into_inner();
        let _ = stream
            .write_all(br#"{"success":true,"data":{"ok":true}}"#)
            .await;
        let _ = stream.write_all(b"\n").await;
        let _ = tx.send(command);
    });

    rx
}

// ---------------------------------------------------------------------------
// Core: launch, navigate, evaluate, url, title, close
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_launch_navigate_evaluate_close() {
    let mut state = DaemonState::new();

    // Launch headless Chrome
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["launched"], true);

    // Navigate to example.com
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["url"], "https://example.com/");
    assert_eq!(get_data(&resp)["title"], "Example Domain");

    // Get URL
    let resp = execute_command(&json!({ "id": "3", "action": "url" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["url"], "https://example.com/");

    // Get title
    let resp = execute_command(&json!({ "id": "4", "action": "title" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["title"], "Example Domain");

    // Evaluate JS
    let resp = execute_command(
        &json!({ "id": "5", "action": "evaluate", "script": "1 + 2" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], 3);

    // Evaluate document.title
    let resp = execute_command(
        &json!({ "id": "6", "action": "evaluate", "script": "document.title" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "Example Domain");

    // Close
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["closed"], true);
}

#[tokio::test]
#[ignore]
async fn e2e_lightpanda_launch_can_open_page() {
    let lightpanda_bin = match std::env::var("LIGHTPANDA_BIN") {
        Ok(path) if !path.is_empty() => path,
        _ => return,
    };

    let mut state = DaemonState::new();

    let resp = tokio::time::timeout(
        tokio::time::Duration::from_secs(20),
        execute_command(
            &json!({
                "id": "1",
                "action": "launch",
                "headless": true,
                "engine": "lightpanda",
                "executablePath": lightpanda_bin,
            }),
            &mut state,
        ),
    )
    .await
    .expect("Lightpanda launch should not hang");

    assert_success(&resp);
    assert_eq!(get_data(&resp)["launched"], true);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["url"], "https://example.com/");
    assert_eq!(get_data(&resp)["title"], "Example Domain");

    let resp = execute_command(&json!({ "id": "3", "action": "close" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["closed"], true);
}

#[tokio::test]
#[ignore]
async fn e2e_lightpanda_auto_launch_can_open_page() {
    let lightpanda_bin = match std::env::var("LIGHTPANDA_BIN") {
        Ok(path) if !path.is_empty() => path,
        _ => return,
    };

    let prev_engine = std::env::var("AGENT_BROWSER_ENGINE").ok();
    let prev_path = std::env::var("AGENT_BROWSER_EXECUTABLE_PATH").ok();
    std::env::set_var("AGENT_BROWSER_ENGINE", "lightpanda");
    std::env::set_var("AGENT_BROWSER_EXECUTABLE_PATH", &lightpanda_bin);

    let mut state = DaemonState::new();

    let resp = tokio::time::timeout(
        tokio::time::Duration::from_secs(20),
        execute_command(
            &json!({ "id": "1", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        ),
    )
    .await
    .expect("Lightpanda auto-launch should not hang");

    match prev_engine {
        Some(value) => std::env::set_var("AGENT_BROWSER_ENGINE", value),
        None => std::env::remove_var("AGENT_BROWSER_ENGINE"),
    }
    match prev_path {
        Some(value) => std::env::set_var("AGENT_BROWSER_EXECUTABLE_PATH", value),
        None => std::env::remove_var("AGENT_BROWSER_EXECUTABLE_PATH"),
    }

    assert_success(&resp);
    assert_eq!(get_data(&resp)["url"], "https://example.com/");
    assert_eq!(get_data(&resp)["title"], "Example Domain");

    let resp = execute_command(&json!({ "id": "2", "action": "close" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["closed"], true);
}

#[tokio::test]
async fn test_obscura_launch_uses_request_proxy_bypass() {
    let env = EnvGuard::new(&["AGENT_BROWSER_PROXY_BYPASS"]);
    env.set("AGENT_BROWSER_PROXY_BYPASS", "stale-daemon-value");
    for (options, expected) in [
        (json!({"proxyBypass": "localhost"}), "--proxy-bypass"),
        (json!({"proxyBypass": null}), "Failed to launch Obscura"),
        (json!({"proxy": {"bypass": "localhost"}}), "--proxy-bypass"),
        (json!({}), "Failed to launch Obscura"),
    ] {
        let mut state = DaemonState::new();
        let mut request = json!({
            "id": "1", "action": "launch", "headless": true,
            "engine": "obscura", "executablePath": "/nonexistent/obscura",
        });
        request
            .as_object_mut()
            .unwrap()
            .extend(options.as_object().unwrap().clone());
        let resp = execute_command(&request, &mut state).await;
        assert_eq!(resp["success"], false, "{resp}");
        assert!(
            resp["error"]
                .as_str()
                .is_some_and(|error| error.contains(expected)),
            "{resp}"
        );
    }
}

fn required_obscura_binary() -> String {
    let path = std::env::var("OBSCURA_BIN")
        .expect("OBSCURA_BIN is required for explicitly invoked Obscura E2E verification");
    assert!(!path.trim().is_empty(), "OBSCURA_BIN must not be empty");
    assert!(
        std::path::Path::new(&path).is_file(),
        "OBSCURA_BIN must point to an existing executable: {path}"
    );
    path
}

struct ObscuraFixture {
    url: String,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for ObscuraFixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn obscura_fixture() -> ObscuraFixture {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request).await;
            let body = include_str!("test_fixtures/obscura_probe.html");
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            let _ = stream.write_all(response.as_bytes()).await;
        }
    });
    ObscuraFixture { url, server }
}

async fn obscura_command(state: &mut DaemonState, command: Value) -> Value {
    tokio::time::timeout(
        tokio::time::Duration::from_secs(30),
        execute_command(&command, state),
    )
    .await
    .expect("Obscura command must not hang")
}

fn obscura_test_environment() -> EnvGuard<'static> {
    let env = EnvGuard::new(&[
        "AGENT_BROWSER_CDP",
        "AGENT_BROWSER_AUTO_CONNECT",
        "AGENT_BROWSER_PROVIDER",
        "AGENT_BROWSER_ENGINE",
        "AGENT_BROWSER_EXECUTABLE_PATH",
        "OBSCURA_ALLOW_PRIVATE_NETWORK",
        "AGENT_BROWSER_OBSCURA_STEALTH",
        "AGENT_BROWSER_PROXY",
        "AGENT_BROWSER_PROXY_BYPASS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ]);
    for key in [
        "AGENT_BROWSER_CDP",
        "AGENT_BROWSER_AUTO_CONNECT",
        "AGENT_BROWSER_PROVIDER",
        "AGENT_BROWSER_ENGINE",
        "AGENT_BROWSER_EXECUTABLE_PATH",
        "AGENT_BROWSER_OBSCURA_STEALTH",
        "AGENT_BROWSER_PROXY",
        "AGENT_BROWSER_PROXY_BYPASS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ] {
        env.remove(key);
    }
    env.set("OBSCURA_ALLOW_PRIVATE_NETWORK", "1");
    env
}

async fn verify_obscura_page(auto_launch: bool) {
    let obscura_bin = required_obscura_binary();
    let env = obscura_test_environment();
    let fixture = obscura_fixture().await;
    let mut state = DaemonState::new();
    if auto_launch {
        env.set("AGENT_BROWSER_ENGINE", "obscura");
        env.set("AGENT_BROWSER_EXECUTABLE_PATH", &obscura_bin);
    } else {
        let resp = obscura_command(
            &mut state,
            json!({
                "id": "1", "action": "launch", "headless": true,
                "engine": "obscura", "executablePath": obscura_bin,
            }),
        )
        .await;
        assert_success(&resp);
        assert_eq!(get_data(&resp)["launched"], true);
    }
    let resp = obscura_command(
        &mut state,
        json!({
            "id": "2", "action": "navigate", "url": fixture.url,
        }),
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["url"], fixture.url);
    assert_eq!(get_data(&resp)["title"], "Obscura adapter probe");
    assert_eq!(state.engine, "obscura");
    assert!(
        state
            .browser
            .as_ref()
            .is_some_and(|manager| manager.owns_obscura_process()),
        "Obscura verification must own an Obscura process, not an attached browser"
    );

    let resp = obscura_command(&mut state, json!({
        "id": "3", "action": "evaluate", "script": "document.querySelector('#message').textContent",
    })).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "JavaScript ready");

    let resp = obscura_command(
        &mut state,
        json!({
            "id": "4", "action": "snapshot",
        }),
    )
    .await;
    assert_success(&resp);
    assert!(
        get_data(&resp)["snapshot"]
            .as_str()
            .is_some_and(|text| text.contains("Adapter probe")),
        "{resp}"
    );

    let resp = obscura_command(&mut state, json!({ "id": "5", "action": "close" })).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["closed"], true);
}

#[test]
fn test_obscura_e2e_clears_inherited_connection_modes() {
    const PROBE: &str = "OBSCURA_TEST_ENV_PROBE";
    let keys = [
        "AGENT_BROWSER_CDP",
        "AGENT_BROWSER_AUTO_CONNECT",
        "AGENT_BROWSER_PROVIDER",
    ];
    if std::env::var(PROBE).as_deref() == Ok("1") {
        let _env = obscura_test_environment();
        for key in keys {
            assert!(
                std::env::var_os(key).is_none(),
                "Obscura E2E must clear {key}"
            );
        }
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .env_clear()
        .env("HOME", dir.path())
        .env(PROBE, "1")
        .env(keys[0], format!("ws://{}", listener.local_addr().unwrap()))
        .env(keys[1], "1")
        .env(keys[2], "obscura-test-not-a-provider")
        .args([
            "--exact",
            "native::e2e_tests::test_obscura_e2e_clears_inherited_connection_modes",
            "--nocapture",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[tokio::test]
async fn test_obscura_e2e_unlaunchable_binary_does_not_attach_to_inherited_cdp() {
    use futures_util::SinkExt;
    use tokio_tungstenite::tungstenite::Message;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("ws://{}", listener.local_addr().unwrap());
    let accepted = Arc::new(AtomicUsize::new(0));
    let server_accepted = accepted.clone();
    let server = tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            server_accepted.fetch_add(1, Ordering::SeqCst);
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            while let Some(Ok(message)) = ws.next().await {
                match message {
                    Message::Text(text) => {
                        let command: Value = serde_json::from_str(&text).unwrap();
                        let response = json!({"id": command["id"], "error": {
                            "code": -32000, "message": "controlled CDP endpoint must not be used by Obscura verification"
                        }});
                        let _ = ws.send(Message::Text(response.to_string())).await;
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        }
    });
    let fixture = ObscuraFixture {
        url: endpoint.clone(),
        server,
    };
    let dir = tempfile::tempdir().unwrap();
    let false_bin = if std::path::Path::new("/usr/bin/false").is_file() {
        "/usr/bin/false"
    } else {
        "/bin/false"
    };
    let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
    command
        .env_clear()
        .env("HOME", dir.path())
        .env("AGENT_BROWSER_SOCKET_DIR", dir.path())
        .env("OBSCURA_BIN", false_bin)
        .env("AGENT_BROWSER_CDP", &fixture.url)
        .env("AGENT_BROWSER_PROVIDER", "obscura-test-not-a-provider")
        .args([
            "--exact",
            "native::e2e_tests::e2e_obscura_auto_launch_can_open_page",
            "--ignored",
            "--nocapture",
        ])
        .kill_on_drop(true);
    let output = tokio::time::timeout(std::time::Duration::from_secs(10), command.output())
        .await
        .expect("Obscura false-binary probe must terminate")
        .unwrap();
    let diagnostics = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(101), "{diagnostics}");
    assert!(
        diagnostics.contains("Obscura exited before CDP became ready"),
        "accepted {} inherited CDP connections\n{diagnostics}",
        accepted.load(Ordering::SeqCst)
    );
    assert_eq!(
        accepted.load(Ordering::SeqCst),
        0,
        "Obscura E2E attached to the inherited endpoint"
    );
}

/// Requires a real OBSCURA_BIN; missing binaries fail rather than silently skip verification.
#[tokio::test]
#[ignore = "requires OBSCURA_BIN; run serially"]
async fn e2e_obscura_launch_can_open_page() {
    verify_obscura_page(false).await;
}

#[tokio::test]
#[ignore = "requires OBSCURA_BIN; run serially"]
async fn e2e_obscura_auto_launch_can_open_page() {
    verify_obscura_page(true).await;
}

// ---------------------------------------------------------------------------
// Runtime stream lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_runtime_stream_enable_before_launch_attaches_and_disables() {
    let guard = EnvGuard::new(&["AGENT_BROWSER_SOCKET_DIR", "AGENT_BROWSER_SESSION"]);
    let socket_dir = std::env::temp_dir().join(format!(
        "agent-browser-e2e-stream-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&socket_dir).expect("socket dir should be created");
    guard.set(
        "AGENT_BROWSER_SOCKET_DIR",
        socket_dir.to_str().expect("socket dir should be utf-8"),
    );
    guard.set("AGENT_BROWSER_SESSION", "e2e-runtime-stream");

    let mut state = DaemonState::new();

    let resp = execute_command(&json!({ "id": "1", "action": "stream_status" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["enabled"], false);

    let resp = execute_command(
        &json!({ "id": "2", "action": "stream_enable", "port": 0 }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let port = get_data(&resp)["port"]
        .as_u64()
        .expect("stream enable should report the bound port");
    assert_eq!(get_data(&resp)["connected"], false);

    let stream_path = socket_dir.join("e2e-runtime-stream.stream");
    assert!(
        stream_path.exists(),
        "runtime enable should create .stream metadata"
    );

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .expect("websocket client should connect to runtime stream");

    let initial = tokio::time::timeout(tokio::time::Duration::from_secs(5), ws.next())
        .await
        .expect("websocket should emit initial status")
        .expect("websocket should stay open")
        .expect("websocket message should be valid");
    let initial_text = initial.into_text().expect("initial message should be text");
    let initial_status: Value =
        serde_json::from_str(&initial_text).expect("status JSON should parse");
    assert_eq!(initial_status["type"], "status");
    assert_eq!(initial_status["connected"], false);

    let resp = execute_command(
        &json!({ "id": "3", "action": "navigate", "url": "data:text/html,<h1>Runtime Stream</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let mut observed_connected = false;
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        let Some(message) = tokio::time::timeout(tokio::time::Duration::from_secs(2), ws.next())
            .await
            .expect("websocket should emit status after browser launch")
        else {
            continue;
        };
        let message = message.expect("websocket message should be valid");
        if !message.is_text() {
            continue;
        }
        let parsed: Value =
            serde_json::from_str(message.to_text().expect("text message should be readable"))
                .expect("runtime stream payload should be valid JSON");
        if parsed.get("type") == Some(&json!("status"))
            && parsed.get("connected") == Some(&json!(true))
        {
            observed_connected = true;
            break;
        }
    }
    assert!(
        observed_connected,
        "runtime stream should report connected=true after browser launch"
    );

    let resp = execute_command(
        &json!({ "id": "4", "action": "stream_disable" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["disabled"], true);
    assert!(
        !stream_path.exists(),
        "stream disable should remove .stream metadata"
    );

    let close_message = tokio::time::timeout(tokio::time::Duration::from_secs(5), ws.next())
        .await
        .expect("websocket should close after disable");
    assert!(
        close_message.is_none() || close_message.expect("ws result should exist").is_ok(),
        "websocket should shut down cleanly when the runtime stream is disabled"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    let _ = std::fs::remove_dir_all(&socket_dir);
}

#[cfg(unix)]
#[tokio::test]
#[ignore]
async fn e2e_stream_command_requires_same_origin_before_daemon_relay() {
    let guard = EnvGuard::new(&["AGENT_BROWSER_SOCKET_DIR", "AGENT_BROWSER_SESSION"]);
    let temp_parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("t");
    std::fs::create_dir_all(&temp_parent).expect("socket temp parent should be created");
    let socket_dir = tempfile::Builder::new()
        .prefix("ab-e2e-")
        .tempdir_in(temp_parent)
        .expect("socket dir should be created");
    guard.set(
        "AGENT_BROWSER_SOCKET_DIR",
        socket_dir
            .path()
            .to_str()
            .expect("socket dir should be utf-8"),
    );
    guard.set("AGENT_BROWSER_SESSION", "x");

    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "stream_enable", "port": 0 }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let port = get_data(&resp)["port"]
        .as_u64()
        .expect("stream enable should report the bound port");

    let mut daemon_command = spawn_fake_daemon_socket(socket_dir.path(), "x").await;
    let body = r#"{"action":"tabs"}"#;
    let cross_origin_request = format!(
        "POST /api/command HTTP/1.1\r\nHost: localhost:{port}\r\nOrigin: https://evil.example\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_raw_http_request(port, &cross_origin_request).await;
    assert!(
        response.starts_with("HTTP/1.1 403 Forbidden"),
        "unexpected cross-origin response: {response}"
    );
    assert!(
        !response.contains("Access-Control-Allow-Origin: *"),
        "forbidden command response exposed wildcard CORS: {response}"
    );
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), &mut daemon_command)
            .await
            .is_err(),
        "cross-origin command request reached daemon relay"
    );

    let same_origin_request = format!(
        "POST /api/command HTTP/1.1\r\nHost: localhost:{port}\r\nOrigin: http://localhost:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let response = send_raw_http_request(port, &same_origin_request).await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "unexpected same-origin response: {response}"
    );
    assert!(
        response.contains(&format!(
            "Access-Control-Allow-Origin: http://localhost:{port}"
        )),
        "same-origin command response did not reflect origin: {response}"
    );
    assert!(
        !response.contains("Access-Control-Allow-Origin: *"),
        "same-origin command response exposed wildcard CORS: {response}"
    );

    let relayed = tokio::time::timeout(std::time::Duration::from_secs(1), daemon_command)
        .await
        .expect("same-origin request should reach fake daemon")
        .expect("fake daemon should return relayed command");
    assert!(relayed.contains(r#""action":"tabs""#), "{relayed}");

    let resp = execute_command(
        &json!({ "id": "2", "action": "stream_disable" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Snapshot with refs and ref-based click
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_snapshot_and_click_ref() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Take snapshot
    let resp = execute_command(&json!({ "id": "3", "action": "snapshot" }), &mut state).await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap();
    assert!(
        snapshot.contains("Example Domain"),
        "Snapshot should contain heading"
    );
    assert!(snapshot.contains("ref=e1"), "Snapshot should have ref e1");
    assert!(snapshot.contains("ref=e2"), "Snapshot should have ref e2");
    assert!(
        snapshot.contains("link"),
        "Snapshot should have a link element"
    );

    // Click the link by ref (e2 is the "More information..." link)
    let resp = execute_command(
        &json!({ "id": "4", "action": "click", "selector": "e2" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Wait for navigation
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Verify URL changed
    let resp = execute_command(&json!({ "id": "5", "action": "url" }), &mut state).await;
    assert_success(&resp);
    let url = get_data(&resp)["url"].as_str().unwrap();
    assert!(
        url.contains("iana.org"),
        "Should have navigated to iana.org, got: {}",
        url
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_snapshot_refs_survive_dom_updates_and_never_recycle() {
    let mut state = DaemonState::new();
    assert_success(
        &execute_command(
            &json!({ "id": "1", "action": "launch", "headless": true }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "2", "action": "navigate", "url": "about:blank" }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "3", "action": "setcontent", "html": "<button id='a'>Alpha</button><button id='b'>Beta</button>" }),
            &mut state,
        )
        .await,
    );

    let first = execute_command(&json!({ "id": "4", "action": "snapshot" }), &mut state).await;
    assert_success(&first);
    let alpha_ref = get_data(&first)["refs"]
        .as_object()
        .unwrap()
        .iter()
        .find(|(_, node)| node["name"] == "Alpha")
        .map(|(ref_id, _)| ref_id.clone())
        .unwrap();

    assert_success(
        &execute_command(
            &json!({ "id": "5", "action": "evaluate", "script": "document.body.prepend(document.getElementById('a')); document.getElementById('b').remove()" }),
            &mut state,
        )
        .await,
    );
    let second = execute_command(&json!({ "id": "6", "action": "snapshot" }), &mut state).await;
    assert_success(&second);
    assert_eq!(get_data(&second)["refs"][&alpha_ref]["name"], "Alpha");
    assert_eq!(
        get_data(&second)["removedRefs"].as_array().unwrap().len(),
        1
    );

    assert_success(
        &execute_command(
            &json!({ "id": "7", "action": "navigate", "url": "about:blank?new-document" }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "8", "action": "setcontent", "html": "<button>Alpha</button>" }),
            &mut state,
        )
        .await,
    );
    let third = execute_command(&json!({ "id": "9", "action": "snapshot" }), &mut state).await;
    assert_success(&third);
    assert!(get_data(&third)["refs"].get(&alpha_ref).is_none());
    assert_success(&execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await);
}

#[tokio::test]
#[ignore]
async fn e2e_snapshot_refs_invalidate_iframe_navigation() {
    let mut state = DaemonState::new();
    for command in [
        json!({"action": "launch", "headless": true}),
        json!({"action": "navigate", "url": "about:blank"}),
        json!({"action": "setcontent", "html": "<button>Parent</button><iframe id='child'></iframe>"}),
    ] {
        assert_success(&execute_command(&command, &mut state).await);
    }
    let replace = json!({"action": "evaluate", "script": "new Promise(resolve => { const f = document.getElementById('child'); f.onload = () => resolve(true); f.srcdoc = '<button>Child</button>'; })"});
    assert_success(&execute_command(&replace, &mut state).await);
    let first = execute_command(&json!({"action": "snapshot"}), &mut state).await;
    assert_success(&first);
    let find_ref = |snapshot: &Value, name: &str| {
        get_data(snapshot)["refs"]
            .as_object()
            .unwrap()
            .iter()
            .find(|(_, node)| node["name"] == name)
            .unwrap()
            .0
            .clone()
    };
    let parent = find_ref(&first, "Parent");
    let child = find_ref(&first, "Child");
    let unchanged = execute_command(&json!({"action": "snapshot"}), &mut state).await;
    assert_eq!(find_ref(&unchanged, "Child"), child);
    assert_success(&execute_command(&replace, &mut state).await);
    let replaced = execute_command(&json!({"action": "snapshot"}), &mut state).await;
    assert_success(&replaced);
    assert_eq!(find_ref(&replaced, "Parent"), parent);
    assert_ne!(find_ref(&replaced, "Child"), child);
    assert!(get_data(&replaced)["removedRefs"]
        .as_array()
        .unwrap()
        .contains(&json!(format!("@{child}"))));
    assert_success(&execute_command(&json!({"action": "close"}), &mut state).await);
}

// ---------------------------------------------------------------------------
// Screenshot
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_screenshot() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Default screenshot
    let resp = execute_command(&json!({ "id": "3", "action": "screenshot" }), &mut state).await;
    assert_success(&resp);
    let path = get_data(&resp)["path"].as_str().unwrap();
    assert!(path.ends_with(".png"), "Screenshot path should be .png");
    let metadata = std::fs::metadata(path).expect("Screenshot file should exist");
    assert!(
        metadata.len() > 1000,
        "Screenshot should be non-trivial size"
    );

    // Named screenshot
    let tmp_path = std::env::temp_dir()
        .join("agent-browser-e2e-test-screenshot.png")
        .to_string_lossy()
        .to_string();
    let resp = execute_command(
        &json!({ "id": "4", "action": "screenshot", "path": tmp_path }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(std::path::Path::new(&tmp_path).exists());
    let _ = std::fs::remove_file(&tmp_path);

    let resp = execute_command(
        &json!({ "id": "4a", "action": "screenshot", "ifChanged": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["changed"], true);
    assert_eq!(get_data(&resp)["revision"], 1);
    let conditional_path = get_data(&resp)["path"].as_str().unwrap().to_string();

    let resp = execute_command(
        &json!({ "id": "4b", "action": "screenshot", "ifChanged": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["changed"], false);
    assert_eq!(get_data(&resp)["revision"], 2);
    assert_eq!(get_data(&resp)["pixelChangeRatio"], 0.0);
    assert!(get_data(&resp).get("path").is_none());
    let _ = std::fs::remove_file(conditional_path);

    let resp = execute_command(
        &json!({
            "id": "5",
            "action": "setcontent",
            "html": r##"
                <html><body>
                  <button onclick="document.getElementById('result').textContent = 'clicked'">Submit</button>
                  <a href="#">Home</a>
                  <div id="result"></div>
                </body></html>
            "##,
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "5a", "action": "screenshot", "ifChanged": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["changed"], true);
    assert_eq!(get_data(&resp)["revision"], 3);
    assert!(get_data(&resp)["pixelChangeRatio"].as_f64().unwrap() > 0.0);
    let _ = std::fs::remove_file(get_data(&resp)["path"].as_str().unwrap());

    let resp = execute_command(
        &json!({ "id": "6", "action": "screenshot", "annotate": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let annotations = get_data(&resp)["annotations"]
        .as_array()
        .expect("Annotated screenshot should return annotations");
    assert!(
        !annotations.is_empty(),
        "Annotated screenshot should have at least one annotation"
    );

    let submit_ref = annotations
        .iter()
        .find(|ann| ann.get("name").and_then(|v| v.as_str()) == Some("Submit"))
        .and_then(|ann| ann.get("ref").and_then(|v| v.as_str()))
        .expect("Expected a Submit annotation");

    let resp = execute_command(
        &json!({
            "id": "7",
            "action": "evaluate",
            "script": "document.getElementById('__agent_browser_annotations__') === null"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], true);

    let resp = execute_command(
        &json!({ "id": "8", "action": "click", "selector": format!("@{}", submit_ref) }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "9",
            "action": "evaluate",
            "script": "document.getElementById('result').textContent"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "clicked");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Form interaction: fill, type, select, check
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_form_interaction() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = concat!(
        "data:text/html,<html><body>",
        "<input id='name' type='text' placeholder='Name'>",
        "<input id='email' type='email'>",
        "<select id='color'><option value='red'>Red</option><option value='blue'>Blue</option></select>",
        "<input id='agree' type='checkbox'>",
        "<textarea id='bio'></textarea>",
        "<button id='submit'>Submit</button>",
        "</body></html>"
    );

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Fill name
    let resp = execute_command(
        &json!({ "id": "10", "action": "fill", "selector": "#name", "value": "John Doe" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Verify fill
    let resp = execute_command(
        &json!({ "id": "11", "action": "evaluate", "script": "document.getElementById('name').value" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "John Doe");

    // Type email – the type action now correctly handles punctuation like '.'
    let resp = execute_command(
        &json!({ "id": "12", "action": "type", "selector": "#email", "text": "john@example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "13", "action": "evaluate", "script": "document.getElementById('email').value" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "john@example.com");

    // Select option
    let resp = execute_command(
        &json!({ "id": "14", "action": "select", "selector": "#color", "values": ["blue"] }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "15", "action": "evaluate", "script": "document.getElementById('color').value" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "blue");

    // Check checkbox
    let resp = execute_command(
        &json!({ "id": "16", "action": "check", "selector": "#agree" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "17", "action": "ischecked", "selector": "#agree" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["checked"], true);

    // Uncheck
    let resp = execute_command(
        &json!({ "id": "18", "action": "uncheck", "selector": "#agree" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "19", "action": "ischecked", "selector": "#agree" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["checked"], false);

    // Snapshot should show form state
    let resp = execute_command(&json!({ "id": "20", "action": "snapshot" }), &mut state).await;
    assert_success(&resp);
    let snap = get_data(&resp)["snapshot"].as_str().unwrap();
    assert!(
        snap.contains("John Doe"),
        "Snapshot should show filled value"
    );
    assert!(snap.contains("textbox"), "Snapshot should show textbox");
    assert!(snap.contains("button"), "Snapshot should show button");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_fill_sets_value_type_inputs_directly() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = r#"<input id="date" type="date"><input id="time" type="time">
        <input id="datetime" type="datetime-local"><input id="month" type="month">
        <input id="week" type="week"><input id="color" type="color">
        <input id="range" type="range" max="100"><input id="locked" type="date" value="2024-01-01" readonly>
        <script>
            window.events = [];
            for (const el of document.querySelectorAll('input'))
                for (const type of ['input', 'change'])
                    el.addEventListener(type, () => events.push(el.id + ':' + type));
            // React records each value it writes through an instance-level
            // value property and reports a change only when the DOM differs.
            const date = document.getElementById('date');
            const native = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
            let tracked = date.value;
            Object.defineProperty(date, 'value', {
                get() { return native.get.call(this); },
                set(v) { tracked = String(v); native.set.call(this, v); },
            });
            window.reactChanges = [];
            date.addEventListener('input', () => {
                if (date.value !== tracked) { tracked = date.value; reactChanges.push(date.value); }
            });
        </script>"#;
    // Base64, since a plain data: URL ends at the first '#'.
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Issue #2000: Chrome ignores typed text on these, so fill used to leave
    // them empty. Playwright trims the value, and Chrome lowercases colors and
    // normalizes some valid values (seconds of :00, a trailing .0).
    let cases = [
        ("date", "2024-01-15", "2024-01-15"),
        ("time", " 09:30 ", "09:30"),
        ("datetime", "2024-01-15T09:30:00", "2024-01-15T09:30"),
        ("month", "2024-03", "2024-03"),
        ("week", "2024-W05", "2024-W05"),
        ("color", "#FF8800", "#ff8800"),
        ("range", "75.0", "75"),
    ];
    let mut events = Vec::new();
    for (id, value, expected) in cases {
        let resp = execute_command(
            &json!({ "id": id, "action": "fill", "selector": format!("#{id}"), "value": value }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        let script = format!("document.getElementById('{id}').value");
        assert_evaluate(&mut state, id, &script, json!(expected)).await;
        events.extend([format!("{id}:input"), format!("{id}:change")]);
    }
    assert_evaluate(&mut state, "events", "events", json!(events)).await;
    assert_evaluate(&mut state, "react", "reactChanges", json!(["2024-01-15"])).await;

    // A value the browser rejects or clamps (max is 100, so the midpoint is
    // 50) and a readonly input (the value setter ignores readonly) are errors
    // that leave the field alone.
    let refused = [
        ("#date", "01/15/2024", "Malformed value"),
        ("#range", "150", "Malformed value"),
        // Not a number to HTML, so Chrome sets the midpoint, though Number()
        // reads it as 50.
        ("#range", "0x32", "Malformed value"),
        (
            "#locked",
            "2024-02-02",
            "Element '#locked' is readonly and cannot be filled",
        ),
    ];
    for (selector, value, error) in refused {
        let resp = execute_command(
            &json!({ "id": "bad", "action": "fill", "selector": selector, "value": value }),
            &mut state,
        )
        .await;
        assert_eq!(resp["success"], false, "{selector}: {resp}");
        assert!(
            resp["error"].as_str().unwrap_or_default().contains(error),
            "{selector}: {resp}"
        );
    }
    assert_evaluate(
        &mut state,
        "kept",
        "['date', 'range', 'locked'].map((id) => document.getElementById(id).value).concat(events.length)",
        json!(["2024-01-15", "75", "2024-01-01", events.len()]),
    )
    .await;

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_fill_and_type_refuse_elements_that_cannot_take_text() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = concat!(
        "data:text/html,<input id='name'>",
        "<select id='choice'><option>a</option></select>",
        "<input id='agree' type='checkbox'>",
        "<button id='go'>Go</button>",
        "<input id='color' type='color'>",
        "<input id='hidden' style='display:none'>",
        "<div id='ce-box' contenteditable>outer<input id='ce-disabled' disabled value='old'>",
        "<textarea id='ce-hidden' style='display:none'>old</textarea>",
        "<select id='ce-select'><option>a</option></select><button id='ce-button' disabled value='v'>B</button></div>",
        "<div id='nest-box' contenteditable>before <span id='nest' contenteditable tabindex='0'>text</span>",
        " <span id='nest-plain' tabindex='0'>abc</span></div><div id='keep' tabindex='0'>plain</div>",
        "<div id='moves-out' tabindex='0'>out</div><input id='jumps'><x-away id='away' tabindex='0'></x-away>",
        "<div id='svg-box'><svg><slot></slot></svg></div>",
        "<x-form><input id='x-a'><input id='x-b'></x-form><x-open id='open-host'></x-open><div id='closed-host'></div>",
        "<div id='dc-editor' contenteditable style='display: contents'>text<input id='dc-control'></div>",
        "<x-closed-box id='closed-box'></x-closed-box><div id='closed-div-box'></div>",
        "<svg id='svg-fo' width='220' height='40'><foreignObject width='200' height='30'><input id='svg-fo-input'></foreignObject></svg>",
        "<div style='-webkit-user-modify: read-write'>before <span id='css-nest' tabindex='0'>text</span></div>",
        "<label><div id='out-labelled' tabindex='0'>out</div> <input id='out-keep' value='KEEP'></label>",
        "<input id='morph'><input id='morph-date' type='date'><script>",
        "customElements.define('x-open', class extends HTMLElement { constructor() { super();",
        " this.attachShadow({ mode: 'open' }).innerHTML = '<input>'; } });",
        "const hostRoot = document.getElementById('closed-host').attachShadow({ mode: 'closed' });",
        "hostRoot.innerHTML = '<input>'; window.closedHostInput = hostRoot.querySelector('input');",
        "document.getElementById('closed-div-box').attachShadow({ mode: 'closed', delegatesFocus: true })",
        ".innerHTML = '<input type=checkbox>';",
        "customElements.define('x-closed-box', class extends HTMLElement { constructor() { super();",
        " this.attachShadow({ mode: 'closed', delegatesFocus: true }).innerHTML = '<input type=checkbox>'; } });",
        "const toName = () => queueMicrotask(() => document.getElementById('name').focus());",
        "document.getElementById('out-labelled').addEventListener('focus', toName);",
        "for (const id of ['morph', 'morph-date']) { let focuses = 0; const el = document.getElementById(id);",
        " el.addEventListener('focus', () => { if (++focuses === 1) toName(); else el.type = 'checkbox'; }); }",
        "const frame = document.createElement('iframe'); document.body.append(frame);",
        "const adopted = frame.contentDocument.createElement('div'); adopted.id = 'adopted-box';",
        "adopted.innerHTML = '<input id=adopted-input>'; document.body.append(adopted);",
        "document.getElementById('moves-out').addEventListener('focus', toName);",
        "document.getElementById('jumps').addEventListener('focus', toName);",
        "customElements.define('x-away', class extends HTMLElement { constructor() { super();",
        " this.addEventListener('focus', () => setTimeout(() => document.getElementById('name').focus())); } });",
        "</script>",
    );
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "3", "action": "fill", "selector": "#name", "value": "Jane" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Issue #2055: each of these reported success and wrote into #name, the
    // field that last held the selection.
    let not_text = "is not an <input>, <textarea> or [contenteditable] element";
    let checkbox = "of type \"checkbox\" cannot be filled";
    let cases = [
        ("fill", "#choice", not_text),
        ("fill", "#agree", checkbox),
        ("fill", "#go", not_text),
        ("type", "#go", not_text),
        ("type", "#color", "use fill"),
        ("fill", "#hidden", "did not take focus"),
        // Classified by the checkbox inside their closed shadow roots.
        ("fill", "#closed-box", checkbox),
        ("fill", "#closed-div-box", checkbox),
        // A control inside editable content that won't take focus must not
        // hand the text to its editing host.
        ("fill", "#ce-disabled", "did not take focus"),
        ("type", "#ce-hidden", "did not take focus"),
        // Chrome reports isContentEditable for these inside an editing host,
        // but they are refused as anywhere else.
        ("fill", "#ce-select", not_text),
        ("fill", "#ce-button", not_text),
        // A built-in element that keeps focus itself takes no text.
        ("type", "#keep", not_text),
        // Focus a handler moves on in a microtask is read once it has moved,
        // so the text doesn't follow it out into #name.
        ("type", "#moves-out", not_text),
        ("type", "#jumps", "did not take focus"),
        // Focus a timer moves out between reads is not followed out either.
        ("fill", "#away", not_text),
        // An SVG <slot> inside is not a slot that takes assigned nodes.
        ("fill", "#svg-box", not_text),
        // Focus a handler passes out of a host inside a label stays the
        // host's to refuse; it doesn't fall to the label's control.
        ("fill", "#out-labelled", not_text),
        // A handler that makes the target a checkbox when it is focused to
        // take the text, after it was classified, typed or set.
        ("fill", "#morph", "changed before the text reached it"),
        ("fill", "#morph-date", "changed before the text reached it"),
    ];
    for (action, selector, error) in cases {
        let text_key = if action == "fill" { "value" } else { "text" };
        let mut cmd = json!({ "id": action, "action": action, "selector": selector });
        cmd[text_key] = json!("leak");
        let resp = execute_command(&cmd, &mut state).await;
        assert_eq!(resp["success"], false, "{action} {selector}: {resp}");
        assert!(
            resp["error"].as_str().unwrap_or_default().contains(error),
            "{action} {selector}: {resp}"
        );
        assert_evaluate(
            &mut state,
            "name",
            "document.getElementById('name').value",
            json!("Jane"),
        )
        .await;
    }
    assert_evaluate(
        &mut state,
        "untouched",
        "['hidden', 'color', 'ce-disabled', 'ce-hidden', 'ce-select', 'ce-button', 'out-keep', 'morph', 'morph-date']
            .map((id) => document.getElementById(id).value)
            .concat(document.getElementById('ce-box').textContent)",
        json!(["", "#000000", "old", "old", "a", "v", "KEEP", "on", "on", "outeroldaB"]),
    )
    .await;

    // Focusable content inside an editor (a nested editor, or a span with
    // tabindex) takes focus without the caret. A caret already inside it
    // stays; with #name holding the caret, it moves to the end of the element
    // rather than the text landing in #name.
    let focus_name = "document.getElementById('name').focus()";
    let caret_in_span =
        "getSelection().collapse(document.getElementById('nest-plain').firstChild, 1)";
    for (setup, action, key, id, text, expected) in [
        (focus_name, "type", "text", "nest", "X", "textX"),
        (focus_name, "fill", "value", "nest", "Y", "textXY"),
        (caret_in_span, "type", "text", "nest-plain", "X", "aXbc"),
        (focus_name, "type", "text", "nest-plain", "Y", "aXbcY"),
        (focus_name, "fill", "value", "nest-plain", "Z", "aXbcYZ"),
        // Likewise in an editor made editable by -webkit-user-modify.
        (focus_name, "type", "text", "css-nest", "X", "textX"),
    ] {
        let setup = json!({ "id": "nest", "action": "evaluate", "script": setup });
        assert_success(&execute_command(&setup, &mut state).await);
        let selector = format!("#{id}");
        let resp = execute_command(
            &json!({ "id": "nest", "action": action, "selector": selector, key: text }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        let script = format!(
            "[document.getElementById('name').value, document.getElementById('{id}').textContent]"
        );
        assert_evaluate(&mut state, action, &script, json!(["Jane", expected])).await;
    }

    // Focus already inside a container that neither takes focus nor hands it
    // on came from earlier; filling the container must not write there.
    let containers = [
        ("x-form", "document.getElementById('x-a')", not_text),
        (
            "#open-host",
            "document.getElementById('open-host').shadowRoot.querySelector('input')",
            not_text,
        ),
        ("#closed-host", "closedHostInput", not_text),
        // An SVG element whose focus() does nothing, with a foreignObject.
        (
            "#svg-fo",
            "document.getElementById('svg-fo-input')",
            not_text,
        ),
        // Made in a frame, so its prototypes are the frame's, then moved here.
        (
            "#adopted-box",
            "document.getElementById('adopted-input')",
            not_text,
        ),
        (
            "#dc-editor",
            "document.getElementById('dc-control')",
            "did not take focus",
        ),
    ];
    for (selector, field, error) in containers {
        let setup =
            format!("(() => {{ const field = {field}; field.value = 'Ada'; field.focus(); }})()");
        assert_success(
            &execute_command(
                &json!({ "id": "setup", "action": "evaluate", "script": setup }),
                &mut state,
            )
            .await,
        );
        let resp = execute_command(
            &json!({ "id": "fill", "action": "fill", "selector": selector, "value": "Lin" }),
            &mut state,
        )
        .await;
        assert_eq!(resp["success"], false, "{selector}: {resp}");
        assert!(
            resp["error"].as_str().unwrap_or_default().contains(error),
            "{selector}: {resp}"
        );
        assert_evaluate(&mut state, "field", &format!("{field}.value"), json!("Ada")).await;
    }

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_fill_and_type_reach_inputs_behind_labels_custom_elements_and_editors() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = r#"<label>Name <span id="name-text">here</span> <input id="name"></label>
        <label id="when-label" for="when">When</label> <input id="when" type="date">
        <x-field id="field"></x-field>
        <div id="editor" role="textbox" tabindex="0"></div>
        <x-closed id="closed"></x-closed>
        <label id="email-label">Email <fa-field id="email"></fa-field></label>
        <label><x-field id="plain"></x-field> <input id="other"></label>
        <div id="ce-box" contenteditable>outer<input id="inside" value="old"></div>
        <div id="closed-div"></div>
        <label><div id="closed-labelled"></div> <input id="other-labelled"></label>
        <label><div id="label-editor" role="textbox" tabindex="0"></div> <input id="label-other" value="keep"></label>
        <div id="list-editor" contenteditable><ol><li id="list-item">item</li></ol></div>
        <div id="fwd-editor" contenteditable>notes <textarea id="fwd-area"></textarea></div>
        <x-override id="override"></x-override>
        <div id="wrap" role="textbox" tabindex="0"><textarea id="wrap-area"></textarea></div>
        <div id="later" tabindex="0"><input id="later-input"></div>
        <div id="blur-later" tabindex="0"><input id="blur-later-input"></div>
        <div id="css-editor" style="-webkit-user-modify: read-write">hello</div>
        <div id="css-plain" style="-webkit-user-modify: read-write-plaintext-only">plain</div>
        <div id="css-host" style="-webkit-user-modify: read-write"><span id="css-span">span</span></div>
        <input id="css-keep" value="KEEP"><label for="css-keep"><div id="css-labelled" style="-webkit-user-modify: read-write">note</div></label>
        <svg id="svg-override" width="220" height="40"><foreignObject width="200" height="30"><input id="svg-override-input"></foreignObject></svg>
        <svg width="220" height="40"><g id="svg-handler" tabindex="0"><foreignObject width="200" height="30"><input id="svg-handler-input"></foreignObject></g></svg>
        <label><input id="blur-keep" value="KEEP"> <div id="blur-labelled" tabindex="0"><input id="blur-labelled-input"></div></label>
        <div id="frame-wrap" tabindex="0"><iframe id="wrap-frame" srcdoc="<input id='field'>"></iframe></div>
        <x-stop id="stop" tabindex="0"></x-stop>
        <div id="closed-wrap" tabindex="0"></div>
        <x-closed-wrap id="closed-custom" tabindex="0"></x-closed-wrap>
        <div id="deep5" tabindex="0"><input id="deep5-input"></div>
        <div id="deep20" tabindex="0"><input id="deep20-input"></div>
        <script>
            customElements.define('x-field', class extends HTMLElement {
                constructor() {
                    super();
                    this.attachShadow({ mode: 'open', delegatesFocus: true }).innerHTML = '<input>';
                }
            });
            customElements.define('x-closed', class extends HTMLElement {
                constructor() {
                    super();
                    const root = this.attachShadow({ mode: 'closed', delegatesFocus: true });
                    root.innerHTML = '<input>';
                    window.closedInput = root.querySelector('input');
                }
            });
            customElements.define('x-override', class extends HTMLElement {
                constructor() {
                    super();
                    this.attachShadow({ mode: 'open' }).innerHTML = '<input>';
                }
                focus() { this.shadowRoot.querySelector('input').focus(); }
            });
            customElements.define('fa-field', class extends HTMLElement {
                static formAssociated = true;
                constructor() {
                    super();
                    this.attachShadow({ mode: 'open', delegatesFocus: true }).innerHTML = '<input>';
                }
            });
            window.inner = (id) => document.getElementById(id).shadowRoot.querySelector('input').value;
            document.getElementById('wrap').addEventListener('focus', () => {
                document.getElementById('wrap-area').focus();
            });
            document.getElementById('later').addEventListener('focus', () => {
                queueMicrotask(() => document.getElementById('later-input').focus());
            });
            const helper = document.createElement('iframe');
            document.body.append(helper);
            const adopted = helper.contentDocument.createElement('div');
            adopted.id = 'adopted-fwd';
            adopted.tabIndex = 0;
            adopted.innerHTML = '<input id="adopted-fwd-input">';
            document.body.append(adopted);
            adopted.addEventListener('focus', () => document.getElementById('adopted-fwd-input').focus());
            const svgOverride = document.getElementById('svg-override');
            svgOverride.focus = () => document.getElementById('svg-override-input').focus();
            document.getElementById('svg-handler').addEventListener('focus', () => {
                document.getElementById('svg-handler-input').focus();
            });
            for (const id of ['blur-later', 'blur-labelled']) {
                const wrap = document.getElementById(id);
                wrap.addEventListener('focus', () => {
                    wrap.blur();
                    queueMicrotask(() => wrap.querySelector('input').focus());
                });
            }
            window.frameField = () => document.getElementById('wrap-frame').contentDocument.getElementById('field');
            document.getElementById('frame-wrap').addEventListener('focus', () => frameField().focus());
            customElements.define('x-stop', class extends HTMLElement {
                constructor() {
                    super();
                    this.attachShadow({ mode: 'open' }).innerHTML = '<input>';
                    this.addEventListener('focus', (event) => {
                        event.stopImmediatePropagation();
                        this.shadowRoot.querySelector('input').focus();
                    }, true);
                }
            });
            const wrapRoot = document.getElementById('closed-wrap').attachShadow({ mode: 'closed' });
            wrapRoot.innerHTML = '<input>';
            window.closedWrapInput = wrapRoot.querySelector('input');
            document.getElementById('closed-wrap').addEventListener('focus', () => closedWrapInput.focus());
            customElements.define('x-closed-wrap', class extends HTMLElement {
                constructor() {
                    super();
                    const root = this.attachShadow({ mode: 'closed' });
                    root.innerHTML = '<input>';
                    window.closedCustomInput = root.querySelector('input');
                    this.addEventListener('focus', () => closedCustomInput.focus());
                }
            });
            const focusAfter = (id, steps) => document.getElementById(id).addEventListener('focus', async () => {
                for (let i = 0; i < steps; i++) await Promise.resolve();
                document.getElementById(id + '-input').focus();
            });
            focusAfter('deep5', 5);
            focusAfter('deep20', 20);
            const attachClosed = (id) => {
                const root = document.getElementById(id).attachShadow({ mode: 'closed', delegatesFocus: true });
                root.innerHTML = '<input>';
                return root.querySelector('input');
            };
            window.closedDivInput = attachClosed('closed-div');
            window.labelledInput = attachClosed('closed-labelled');
            const labelContext = new EditContext();
            document.getElementById('label-editor').editContext = labelContext;
            window.labelEditorText = '';
            labelContext.addEventListener('textupdate', (e) => { labelEditorText += e.text; });
            document.getElementById('fwd-editor').addEventListener('focus', () => {
                document.getElementById('fwd-area').focus();
            });
            const context = new EditContext();
            document.getElementById('editor').editContext = context;
            window.editorText = '';
            context.addEventListener('textupdate', (e) => { editorText += e.text; });
        </script>"#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // A label (or text inside one) stands for its control, as in Playwright.
    // Custom elements that forward focus to an inner input and EditContext
    // editors (Monaco, VS Code) took typed text before and still do.
    let steps = [
        (
            json!({ "action": "fill", "selector": "#name-text", "value": "Ada" }),
            "document.getElementById('name').value",
            json!("Ada"),
        ),
        (
            json!({ "action": "fill", "selector": "#when-label", "value": "2024-01-15" }),
            "document.getElementById('when').value",
            json!("2024-01-15"),
        ),
        (
            json!({ "action": "fill", "selector": "#field", "value": "Grace" }),
            "inner('field')",
            json!("Grace"),
        ),
        // Filling again while focus is already inside still reaches the input.
        (
            json!({ "action": "fill", "selector": "#field", "value": "Hopper" }),
            "inner('field')",
            json!("Hopper"),
        ),
        (
            json!({ "action": "type", "selector": "#editor", "text": "hi" }),
            "editorText",
            json!("hi"),
        ),
        // Page JS can't see into a closed shadow root; CDP can.
        (
            json!({ "action": "fill", "selector": "#closed", "value": "Hopper" }),
            "closedInput.value",
            json!("Hopper"),
        ),
        (
            json!({ "action": "fill", "selector": "#closed", "value": "Grace" }),
            "closedInput.value",
            json!("Grace"),
        ),
        // A form-associated custom element is itself a label's control.
        (
            json!({ "action": "fill", "selector": "#email", "value": "ada@example.com" }),
            "inner('email')",
            json!("ada@example.com"),
        ),
        (
            json!({ "action": "fill", "selector": "#email-label", "value": "grace@example.com" }),
            "inner('email')",
            json!("grace@example.com"),
        ),
        // A custom element that takes focus keeps the text, even inside a
        // label whose control is another field.
        (
            json!({ "action": "fill", "selector": "#plain", "value": "Lin" }),
            "[inner('plain'), document.getElementById('other').value]",
            json!(["Lin", ""]),
        ),
        // A form control inside editable content takes focus itself; its
        // editing host must not.
        (
            json!({ "action": "fill", "selector": "#inside", "value": "new" }),
            "[document.getElementById('inside').value, document.getElementById('ce-box').textContent]",
            json!(["new", "outer"]),
        ),
        (
            json!({ "action": "type", "selector": "#inside", "text": "Z" }),
            "[document.getElementById('inside').value, document.getElementById('ce-box').textContent]",
            json!(["newZ", "outer"]),
        ),
        // A built-in element can delegate focus into a closed shadow root too.
        (
            json!({ "action": "fill", "selector": "#closed-div", "value": "Ada" }),
            "closedDivInput.value",
            json!("Ada"),
        ),
        (
            json!({ "action": "type", "selector": "#closed-div", "text": "!" }),
            "closedDivInput.value",
            json!("Ada!"),
        ),
        (
            json!({ "action": "fill", "selector": "#closed-labelled", "value": "Lin" }),
            "[labelledInput.value, document.getElementById('other-labelled').value]",
            json!(["Lin", ""]),
        ),
        // An EditContext editor is a text target itself, inside a label too.
        (
            json!({ "action": "fill", "selector": "#label-editor", "value": "REPLACE" }),
            "[labelEditorText, document.getElementById('label-other').value]",
            json!(["REPLACE", "keep"]),
        ),
        // Clearing applies to inputs and textareas only: `value = ''` on an
        // <li> sets value="0" and renumbers the list.
        (
            json!({ "action": "fill", "selector": "#list-item", "value": "X" }),
            "[document.getElementById('list-item').getAttribute('value'),
              document.getElementById('list-editor').textContent.includes('X')]",
            json!([null, true]),
        ),
        // An editing host that moves focus to a control inside it stands for
        // that control, like a shadow host.
        (
            json!({ "action": "type", "selector": "#fwd-editor", "text": "text" }),
            "[document.getElementById('fwd-area').value, document.getElementById('fwd-editor').textContent]",
            json!(["text", "notes "]),
        ),
        (
            json!({ "action": "type", "selector": "#fwd-editor", "text": "!" }),
            "[document.getElementById('fwd-area').value, document.getElementById('fwd-editor').textContent]",
            json!(["text!", "notes "]),
        ),
        // A component that forwards focus from its own focus() keeps working
        // when focus is already inside.
        (
            json!({ "action": "fill", "selector": "#override", "value": "one" }),
            "inner('override')",
            json!("one"),
        ),
        (
            json!({ "action": "fill", "selector": "#override", "value": "two" }),
            "inner('override')",
            json!("two"),
        ),
        // A built-in element whose focus handler passes focus to a control
        // inside it stands for that control.
        (
            json!({ "action": "type", "selector": "#wrap", "text": "hello" }),
            "document.getElementById('wrap-area').value",
            json!("hello"),
        ),
        (
            json!({ "action": "type", "selector": "#wrap", "text": "!" }),
            "document.getElementById('wrap-area').value",
            json!("hello!"),
        ),
        // So does one that passes focus on in a microtask, into a same-origin
        // frame, or after stopping the event; the second time, focus is
        // already there.
        (
            json!({ "action": "type", "selector": "#later", "text": "A" }),
            "document.getElementById('later-input').value",
            json!("A"),
        ),
        (
            json!({ "action": "type", "selector": "#later", "text": "B" }),
            "document.getElementById('later-input').value",
            json!("AB"),
        ),
        // Or that blurs itself first, so focus is on the body when focus()
        // returns, in a label whose control is another field too.
        (
            json!({ "action": "fill", "selector": "#blur-later", "value": "X" }),
            "document.getElementById('blur-later-input').value",
            json!("X"),
        ),
        (
            json!({ "action": "type", "selector": "#blur-later", "text": "Y" }),
            "document.getElementById('blur-later-input').value",
            json!("XY"),
        ),
        (
            json!({ "action": "fill", "selector": "#blur-labelled", "value": "X" }),
            "[document.getElementById('blur-labelled-input').value, document.getElementById('blur-keep').value]",
            json!(["X", "KEEP"]),
        ),
        (
            json!({ "action": "type", "selector": "#blur-labelled", "text": "Y" }),
            "[document.getElementById('blur-labelled-input').value, document.getElementById('blur-keep').value]",
            json!(["XY", "KEEP"]),
        ),
        // Content Chrome edits through -webkit-user-modify, which leaves
        // isContentEditable false, takes text like contenteditable, inside a
        // label for another field too. Focus puts the caret at the start.
        (
            json!({ "action": "fill", "selector": "#css-editor", "value": "X" }),
            "document.getElementById('css-editor').textContent",
            json!("Xhello"),
        ),
        (
            json!({ "action": "type", "selector": "#css-editor", "text": "Y" }),
            "document.getElementById('css-editor').textContent",
            json!("XYhello"),
        ),
        (
            json!({ "action": "fill", "selector": "#css-plain", "value": "X" }),
            "document.getElementById('css-plain').textContent",
            json!("Xplain"),
        ),
        // Content inside one that can't take focus hands it to the editor.
        (
            json!({ "action": "fill", "selector": "#css-span", "value": "X" }),
            "document.getElementById('css-host').textContent",
            json!("Xspan"),
        ),
        (
            json!({ "action": "fill", "selector": "#css-labelled", "value": "X" }),
            "[document.getElementById('css-labelled').textContent, document.getElementById('css-keep').value]",
            json!(["Xnote", "KEEP"]),
        ),
        (
            json!({ "action": "type", "selector": "#css-labelled", "text": "Y" }),
            "[document.getElementById('css-labelled').textContent, document.getElementById('css-keep').value]",
            json!(["XYnote", "KEEP"]),
        ),
        // So does one made in a frame and moved into this document, whose
        // prototypes are the frame's.
        (
            json!({ "action": "fill", "selector": "#adopted-fwd", "value": "X" }),
            "document.getElementById('adopted-fwd-input').value",
            json!("X"),
        ),
        (
            json!({ "action": "type", "selector": "#adopted-fwd", "text": "Y" }),
            "document.getElementById('adopted-fwd-input').value",
            json!("XY"),
        ),
        // An SVG element passes focus on by its own focus() or a handler; its
        // native focus() is SVG's, not HTML's.
        (
            json!({ "action": "fill", "selector": "#svg-override", "value": "X" }),
            "document.getElementById('svg-override-input').value",
            json!("X"),
        ),
        (
            json!({ "action": "type", "selector": "#svg-override", "text": "Y" }),
            "document.getElementById('svg-override-input').value",
            json!("XY"),
        ),
        (
            json!({ "action": "fill", "selector": "#svg-handler", "value": "X" }),
            "document.getElementById('svg-handler-input').value",
            json!("X"),
        ),
        (
            json!({ "action": "type", "selector": "#svg-handler", "text": "Y" }),
            "document.getElementById('svg-handler-input').value",
            json!("XY"),
        ),
        (
            json!({ "action": "type", "selector": "#frame-wrap", "text": "A" }),
            "frameField().value",
            json!("A"),
        ),
        (
            json!({ "action": "type", "selector": "#frame-wrap", "text": "B" }),
            "frameField().value",
            json!("AB"),
        ),
        (
            json!({ "action": "type", "selector": "#stop", "text": "A" }),
            "inner('stop')",
            json!("A"),
        ),
        (
            json!({ "action": "type", "selector": "#stop", "text": "B" }),
            "inner('stop')",
            json!("AB"),
        ),
        // So does a host, built-in or custom, whose focus handler passes focus
        // into its closed shadow root, again when focus is already inside.
        (
            json!({ "action": "fill", "selector": "#closed-wrap", "value": "one" }),
            "closedWrapInput.value",
            json!("one"),
        ),
        (
            json!({ "action": "fill", "selector": "#closed-wrap", "value": "two" }),
            "closedWrapInput.value",
            json!("two"),
        ),
        (
            json!({ "action": "type", "selector": "#closed-wrap", "text": "!" }),
            "closedWrapInput.value",
            json!("two!"),
        ),
        (
            json!({ "action": "fill", "selector": "#closed-custom", "value": "one" }),
            "closedCustomInput.value",
            json!("one"),
        ),
        (
            json!({ "action": "fill", "selector": "#closed-custom", "value": "two" }),
            "closedCustomInput.value",
            json!("two"),
        ),
        (
            json!({ "action": "type", "selector": "#closed-custom", "text": "!" }),
            "closedCustomInput.value",
            json!("two!"),
        ),
        // However deep the promise chain before the handler passes focus on.
        (
            json!({ "action": "fill", "selector": "#deep5", "value": "Ada" }),
            "document.getElementById('deep5-input').value",
            json!("Ada"),
        ),
        (
            json!({ "action": "type", "selector": "#deep5", "text": "!" }),
            "document.getElementById('deep5-input').value",
            json!("Ada!"),
        ),
        (
            json!({ "action": "fill", "selector": "#deep20", "value": "Ada" }),
            "document.getElementById('deep20-input').value",
            json!("Ada"),
        ),
        (
            json!({ "action": "type", "selector": "#deep20", "text": "!" }),
            "document.getElementById('deep20-input').value",
            json!("Ada!"),
        ),
    ];
    for (mut cmd, script, expected) in steps {
        cmd["id"] = json!("step");
        let resp = execute_command(&cmd, &mut state).await;
        assert_success(&resp);
        assert_evaluate(&mut state, "check", script, expected).await;
    }

    // A page whose setTimeout never fires (a fake clock) doesn't stall fill
    // or type, and focus passed on in a microtask is still followed.
    let stub =
        json!({ "id": "stub", "action": "evaluate", "script": "window.setTimeout = () => 0" });
    assert_success(&execute_command(&stub, &mut state).await);
    let stubbed = [
        (
            json!({ "action": "fill", "selector": "#name", "value": "Lin" }),
            "document.getElementById('name').value",
            json!("Lin"),
        ),
        (
            json!({ "action": "type", "selector": "#later", "text": "C" }),
            "document.getElementById('later-input').value",
            json!("ABC"),
        ),
    ];
    for (mut cmd, script, expected) in stubbed {
        cmd["id"] = json!("stubbed");
        let resp = execute_command(&cmd, &mut state).await;
        assert_success(&resp);
        assert_evaluate(&mut state, "stubbed", script, expected).await;
    }

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_type_into_a_display_contents_editor_needs_the_caret_inside() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // With no box the editor can't take focus, but a caret inside it still
    // receives typed text.
    let html = r#"<div id="editor" contenteditable style="display: contents">text</div>
        <input id="other" value="keep"><iframe id="frame" srcdoc="<input id='inner'>"></iframe>"#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let observe =
        "[document.getElementById('editor').textContent, document.getElementById('other').value,
        document.getElementById('frame').contentDocument.getElementById('inner').value]";

    let caret_inside = "(() => { const text = document.getElementById('editor').firstChild; getSelection().collapse(text, text.length); })()";
    assert_success(
        &execute_command(
            &json!({ "id": "3", "action": "evaluate", "script": caret_inside }),
            &mut state,
        )
        .await,
    );
    let resp = execute_command(
        &json!({ "id": "4", "action": "type", "selector": "#editor", "text": "!" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_evaluate(&mut state, "5", observe, json!(["text!", "keep", ""])).await;

    // With focus in another field the text would land there instead. A
    // focused frame keeps this document's caret in the editor, so the caret
    // alone is not enough.
    let elsewhere = [
        "document.getElementById('other').focus()",
        "(() => { const text = document.getElementById('editor').firstChild; getSelection().collapse(text, text.length);
            document.getElementById('frame').contentDocument.getElementById('inner').focus(); })()",
    ];
    for script in elsewhere {
        assert_success(
            &execute_command(
                &json!({ "id": "6", "action": "evaluate", "script": script }),
                &mut state,
            )
            .await,
        );
        let resp = execute_command(
            &json!({ "id": "7", "action": "type", "selector": "#editor", "text": "?" }),
            &mut state,
        )
        .await;
        assert_eq!(resp["success"], false, "{script}: {resp}");
        assert!(
            resp["error"]
                .as_str()
                .unwrap_or_default()
                .contains("did not take focus"),
            "{script}: {resp}"
        );
        assert_evaluate(&mut state, "8", observe, json!(["text!", "keep", ""])).await;
    }

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_fill_and_type_keep_the_caret_in_shadow_root_editors() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Editors in shadow roots: delegatesFocus components reached through
    // their host, and an editor and a display: contents editor reached by
    // ref. A caret placed in a shadow tree reads through
    // document.getSelection() as the host's position, as after a click.
    let html = r#"<input id="other">
        <x-df id="df-ed"></x-df><x-df id="df-fill"></x-df>
        <x-open id="open-ed"></x-open><x-dc id="dc-host"></x-dc>
        <iframe id="frame" srcdoc="<input id='field'>"></iframe>
        <script>
            const editor = (label, style) =>
                `<div id="inner" role="textbox" contenteditable aria-label="${label}" style="${style}">hello world<br></div>`;
            customElements.define('x-df', class extends HTMLElement {
                constructor() {
                    super();
                    this.attachShadow({ mode: 'open', delegatesFocus: true }).innerHTML = editor(this.id, '');
                }
            });
            customElements.define('x-open', class extends HTMLElement {
                constructor() {
                    super();
                    this.attachShadow({ mode: 'open' }).innerHTML = editor('Open editor', '');
                }
            });
            customElements.define('x-dc', class extends HTMLElement {
                constructor() {
                    super();
                    this.attachShadow({ mode: 'open' }).innerHTML = editor('Contents editor', 'display: contents');
                }
            });
            window.inner = (id) => document.getElementById(id).shadowRoot.getElementById('inner');
            window.caretIn = (id, offset) => {
                const el = inner(id);
                el.focus();
                el.getRootNode().getSelection().collapse(el.firstChild, offset);
            };
        </script>"#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "3", "action": "snapshot", "interactive": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap_or_default();
    let ref_for = |label: &str| {
        snapshot
            .lines()
            .find(|line| line.contains(&format!("\"{label}\"")))
            .and_then(|line| line.split("[ref=").nth(1))
            .and_then(|rest| rest.split([']', ',']).next())
            .map(|id| format!("@{id}"))
            .unwrap_or_else(|| panic!("no ref for {label} in {snapshot}"))
    };
    let open_editor = ref_for("Open editor");
    let contents_editor = ref_for("Contents editor");

    // With the caret in the middle of the text, typing goes there, not to the
    // end. fill types where focus() puts the caret, here at the start.
    let rows = [
        (
            "caretIn('df-ed', 5)",
            "type",
            "text",
            "#df-ed".to_string(),
            "df-ed",
        ),
        (
            "caretIn('open-ed', 5)",
            "type",
            "text",
            open_editor,
            "open-ed",
        ),
        (
            "caretIn('dc-host', 5)",
            "type",
            "text",
            contents_editor.clone(),
            "dc-host",
        ),
        (
            "document.getElementById('other').focus()",
            "fill",
            "value",
            "#df-fill".to_string(),
            "df-fill",
        ),
    ];
    for (setup, action, key, selector, host) in rows {
        let setup = json!({ "id": "setup", "action": "evaluate", "script": setup });
        assert_success(&execute_command(&setup, &mut state).await);
        let text = if action == "fill" { "Y" } else { "X" };
        let resp = execute_command(
            &json!({ "id": host, "action": action, "selector": selector, key: text }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        let expected = if action == "fill" {
            "Yhello world<br>"
        } else {
            "helloX world<br>"
        };
        let script = format!("inner('{host}').innerHTML");
        assert_evaluate(&mut state, host, &script, json!(expected)).await;
    }

    // A focused frame keeps the caret in the shadow tree, but the text would
    // go to the frame's field.
    let caret_then_frame = "caretIn('dc-host', 5); \
        document.getElementById('frame').contentDocument.getElementById('field').focus()";
    let setup = json!({ "id": "setup", "action": "evaluate", "script": caret_then_frame });
    assert_success(&execute_command(&setup, &mut state).await);
    let resp = execute_command(
        &json!({ "id": "frame", "action": "type", "selector": contents_editor, "text": "?" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false, "{resp}");
    assert_evaluate(
        &mut state,
        "frame",
        "[inner('dc-host').innerHTML, document.getElementById('frame').contentDocument.getElementById('field').value]",
        json!(["helloX world<br>", ""]),
    )
    .await;

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_fill_and_type_follow_focus_from_shadow_textboxes() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Textboxes inside shadow trees, reached by ref, whose focus handlers pass
    // focus on: to an input slotted in from the host's light DOM, in an open or
    // a closed shadow root, from a textbox or from the slot itself (each once
    // inside a label whose control holds KEEP), or to an input of their own in
    // a closed shadow root. Last, hosts that the page's own capture listener,
    // on the document or added to the window at load, passes focus on from
    // before stopping the event: into their shadow input, or for a light-DOM
    // host back to its slotted input, whose own focus events it stops too.
    // A page that stops blur and focusout as well leaves only whether the host
    // can take focus at all, read before its focus handler can drop its
    // tabindex. Then editors that are display: contents in a shadow root, with
    // the caret in light-DOM text slotted into them.
    let html = r#"<input id="other">
        <x-slot-box id="slot-box"><input id="slotted"></x-slot-box>
        <x-slot-label id="slot-label"><input id="labelled-slotted"></x-slot-label>
        <x-closed-box></x-closed-box>
        <x-closed-slot-box><input id="closed-slotted"></x-closed-slot-box>
        <x-closed-slot-label><input id="closed-labelled-slotted"></x-closed-slot-label>
        <x-closed-slot-beside><input slot="inside"><input id="beside" slot="beside"></x-closed-slot-beside>
        <x-slot-target><input id="slot-target-input"></x-slot-target>
        <x-slot-target-label><input id="slot-target-labelled"></x-slot-target-label>
        <div id="stopped" tabindex="0"></div>
        <label><div id="stopped-labelled" tabindex="0"></div> <input id="stopped-keep" value="KEEP"></label>
        <div id="win-stopped" tabindex="0"></div>
        <label><div id="win-stopped-labelled" tabindex="0"></div> <input id="win-stopped-keep" value="KEEP"></label>
        <x-light-host id="light-host" tabindex="0"><input id="light-input"></x-light-host>
        <x-guard id="guard-host" tabindex="0"><input id="guard-input"></x-guard>
        <label><input id="guard-keep" value="KEEP"> <x-guard id="guard-labelled" tabindex="0"><input id="guard-labelled-input"></x-guard></label>
        <x-guard id="guard-box"><input id="guard-box-input" value="Ada"></x-guard>
        <x-drop id="drop-host" tabindex="0"><input id="drop-input"></x-drop>
        <label><input id="drop-keep" value="KEEP"> <x-drop id="drop-labelled" tabindex="0"><input id="drop-labelled-input"></x-drop></label>
        <div id="slotted-ce-open" contenteditable>hello</div>
        <div id="slotted-ce-closed" contenteditable>hello</div>
        <script>
            const textbox = (label) => `<div id="box" role="textbox" tabindex="0" aria-label="${label}"><slot></slot></div>`;
            const forward = (root, input) => root.getElementById('box').addEventListener('focus', () => input.focus());
            customElements.define('x-slot-box', class extends HTMLElement {
                constructor() {
                    super();
                    const root = this.attachShadow({ mode: 'open' });
                    root.innerHTML = textbox('Slot box');
                    forward(root, this.querySelector('input'));
                }
            });
            customElements.define('x-slot-label', class extends HTMLElement {
                constructor() {
                    super();
                    const root = this.attachShadow({ mode: 'open' });
                    root.innerHTML = `<label>${textbox('Labelled slot box')} <input id="keep" value="KEEP"></label>`;
                    forward(root, this.querySelector('input'));
                }
            });
            customElements.define('x-closed-box', class extends HTMLElement {
                constructor() {
                    super();
                    const root = this.attachShadow({ mode: 'closed' });
                    root.innerHTML = '<div id="box" role="textbox" tabindex="0" aria-label="Closed box"><input></div>';
                    window.closedBoxInput = root.querySelector('input');
                    forward(root, closedBoxInput);
                }
            });
            window.keeps = {};
            const closedHost = (name, html, input) => customElements.define(name, class extends HTMLElement {
                constructor() {
                    super();
                    const root = this.attachShadow({ mode: 'closed' });
                    root.innerHTML = html;
                    keeps[name] = root.getElementById('keep');
                    forward(root, this.querySelector(input));
                }
            });
            closedHost('x-closed-slot-box', textbox('Closed slot box'), 'input');
            closedHost('x-closed-slot-label', `<label>${textbox('Labelled closed slot box')} <input id="keep" value="KEEP"></label>`, 'input');
            // Its textbox passes focus to an input slotted beside it, not into it.
            closedHost('x-closed-slot-beside', '<div id="box" role="textbox" tabindex="0" aria-label="Beside slot box"><slot name="inside"></slot></div><slot name="beside"></slot>', '#beside');
            // A slot with a box takes focus; a title names it for the snapshot.
            const slotTarget = (title) => `<slot id="box" role="textbox" tabindex="0" style="display: block" title="${title}"></slot>`;
            closedHost('x-slot-target', slotTarget('Slot target'), 'input');
            closedHost('x-slot-target-label', `<label>${slotTarget('Labelled slot target')} <input id="keep" value="KEEP"></label>`, 'input');
            const stopped = ['stopped', 'stopped-labelled'];
            for (const id of stopped) document.getElementById(id).attachShadow({ mode: 'open' }).innerHTML = '<input>';
            window.stoppedInput = (id) => document.getElementById(id).shadowRoot.querySelector('input');
            document.addEventListener('focus', (event) => {
                const host = event.composedPath()[0];
                if (!stopped.includes(host.id)) return;
                stoppedInput(host.id).focus();
                event.stopImmediatePropagation();
            }, true);
            const winStopped = ['win-stopped', 'win-stopped-labelled'];
            for (const id of winStopped) document.getElementById(id).attachShadow({ mode: 'open' }).innerHTML = '<input>';
            customElements.define('x-light-host', class extends HTMLElement {
                constructor() {
                    super();
                    this.attachShadow({ mode: 'open' }).innerHTML = '<slot></slot>';
                }
            });
            addEventListener('focus', (event) => {
                const target = event.composedPath()[0];
                if (target.id === 'light-input') return event.stopImmediatePropagation();
                const to = target.id === 'light-host' ? document.getElementById('light-input')
                    : winStopped.includes(target.id) ? stoppedInput(target.id) : null;
                if (!to) return;
                to.focus();
                event.stopImmediatePropagation();
            }, true);
            class Guard extends HTMLElement {
                constructor() {
                    super();
                    this.attachShadow({ mode: 'open' }).innerHTML = '<slot></slot>';
                }
            }
            customElements.define('x-guard', Guard);
            // x-drop also drops its tabindex as it passes focus on, as a roving
            // focus widget does; startInside puts it back and focuses the input
            // directly.
            customElements.define('x-drop', class extends Guard {});
            const guarded = ['guard-host', 'guard-input', 'guard-labelled', 'guard-labelled-input', 'guard-box', 'guard-box-input',
                'drop-host', 'drop-input', 'drop-labelled', 'drop-labelled-input'];
            for (const type of ['focus', 'blur', 'focusout']) addEventListener(type, (event) => {
                const target = event.composedPath()[0];
                if (!guarded.includes(target.id)) return;
                if (type === 'focus' && target.matches('x-guard, x-drop')) target.querySelector('input').focus();
                if (type === 'focus' && target.localName === 'x-drop') target.removeAttribute('tabindex');
                event.stopImmediatePropagation();
            }, true);
            window.startInside = (id) => {
                const host = document.getElementById(id);
                host.tabIndex = 0;
                host.querySelector('input').focus();
            };
            const slottedEditor = (label) => `<div role="textbox" contenteditable aria-label="${label}" style="display: contents"><slot></slot></div>`;
            document.getElementById('slotted-ce-open').attachShadow({ mode: 'open' }).innerHTML = slottedEditor('Open slotted editor');
            document.getElementById('slotted-ce-closed').attachShadow({ mode: 'closed' }).innerHTML = slottedEditor('Closed slotted editor');
            window.caretAtEnd = (id) => {
                const host = document.getElementById(id);
                host.focus();
                getSelection().collapse(host.firstChild, host.firstChild.length);
            };
            window.keep = () => document.getElementById('slot-label').shadowRoot.getElementById('keep').value;
        </script>"#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "3", "action": "fill", "selector": "#other", "value": "Jane" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "4", "action": "snapshot", "interactive": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap_or_default();
    let ref_for = |label: &str| {
        snapshot
            .lines()
            .find(|line| line.contains(&format!("\"{label}\"")))
            .and_then(|line| line.split("[ref=").nth(1))
            .and_then(|rest| rest.split([']', ',']).next())
            .map(|id| format!("@{id}"))
            .unwrap_or_else(|| panic!("no ref for {label} in {snapshot}"))
    };

    // Fill, then type with focus already in the input. The text goes where
    // focus went, not to the label's control, and #other keeps its value.
    for (selector, script, expected) in [
        (
            ref_for("Slot box"),
            "document.getElementById('slotted').value",
            "XY",
        ),
        (
            ref_for("Labelled slot box"),
            "[document.getElementById('labelled-slotted').value, keep()].join(' ')",
            "XY KEEP",
        ),
        (ref_for("Closed box"), "closedBoxInput.value", "XY"),
        (
            ref_for("Closed slot box"),
            "document.getElementById('closed-slotted').value",
            "XY",
        ),
        (
            ref_for("Labelled closed slot box"),
            "[document.getElementById('closed-labelled-slotted').value, keeps['x-closed-slot-label'].value].join(' ')",
            "XY KEEP",
        ),
        (
            ref_for("Slot target"),
            "document.getElementById('slot-target-input').value",
            "XY",
        ),
        (
            ref_for("Labelled slot target"),
            "[document.getElementById('slot-target-labelled').value, keeps['x-slot-target-label'].value].join(' ')",
            "XY KEEP",
        ),
        (
            "#stopped".to_string(),
            "stoppedInput('stopped').value",
            "XY",
        ),
        (
            "#stopped-labelled".to_string(),
            "[stoppedInput('stopped-labelled').value, document.getElementById('stopped-keep').value].join(' ')",
            "XY KEEP",
        ),
        (
            "#win-stopped".to_string(),
            "stoppedInput('win-stopped').value",
            "XY",
        ),
        (
            "#win-stopped-labelled".to_string(),
            "[stoppedInput('win-stopped-labelled').value, document.getElementById('win-stopped-keep').value].join(' ')",
            "XY KEEP",
        ),
        (
            "#light-host".to_string(),
            "document.getElementById('light-input').value",
            "XY",
        ),
        (
            "#guard-host".to_string(),
            "document.getElementById('guard-input').value",
            "XY",
        ),
        (
            "#guard-labelled".to_string(),
            "[document.getElementById('guard-labelled-input').value, document.getElementById('guard-keep').value].join(' ')",
            "XY KEEP",
        ),
    ] {
        for (action, key, text) in [("fill", "value", "X"), ("type", "text", "Y")] {
            let resp = execute_command(
                &json!({ "id": action, "action": action, "selector": selector, key: text }),
                &mut state,
            )
            .await;
            assert_success(&resp);
        }
        assert_evaluate(&mut state, &selector, script, json!(expected)).await;
        assert_evaluate(
            &mut state,
            "other",
            "document.getElementById('other').value",
            json!("Jane"),
        )
        .await;
    }

    // An input slotted beside the textbox, outside it, is not where its text
    // goes, closed shadow root or not.
    let resp = execute_command(
        &json!({ "id": "beside", "action": "fill", "selector": ref_for("Beside slot box"), "value": "X" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false, "{resp}");
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or_default()
            .contains("is not an <input>, <textarea> or [contenteditable] element"),
        "{resp}"
    );
    assert_evaluate(
        &mut state,
        "beside",
        "[document.getElementById('beside').value, document.getElementById('other').value]",
        json!(["", "Jane"]),
    )
    .await;

    // With focus starting on the input, a host that drops its tabindex as it
    // passes focus on still took it: the text goes to the input, not to the
    // label's control.
    for (id, script, expected) in [
        ("drop-host", "document.getElementById('drop-input').value", "XY"),
        (
            "drop-labelled",
            "[document.getElementById('drop-labelled-input').value, document.getElementById('drop-keep').value].join(' ')",
            "XY KEEP",
        ),
    ] {
        for (action, key, text) in [("fill", "value", "X"), ("type", "text", "Y")] {
            let setup = json!({ "id": id, "action": "evaluate", "script": format!("startInside('{id}')") });
            assert_success(&execute_command(&setup, &mut state).await);
            let resp = execute_command(
                &json!({ "id": action, "action": action, "selector": format!("#{id}"), key: text }),
                &mut state,
            )
            .await;
            assert_success(&resp);
        }
        assert_evaluate(&mut state, id, script, json!(expected)).await;
        assert_evaluate(
            &mut state,
            "other",
            "document.getElementById('other').value",
            json!("Jane"),
        )
        .await;
    }

    // A container that can't take focus, on the page that stops every focus
    // event, keeps the field already focused inside it.
    let setup = json!({ "id": "box", "action": "evaluate", "script": "document.getElementById('guard-box-input').focus()" });
    assert_success(&execute_command(&setup, &mut state).await);
    let resp = execute_command(
        &json!({ "id": "box", "action": "fill", "selector": "#guard-box", "value": "X" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false, "{resp}");
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or_default()
            .contains("is not an <input>, <textarea> or [contenteditable] element"),
        "{resp}"
    );
    assert_evaluate(
        &mut state,
        "box",
        "document.getElementById('guard-box-input').value",
        json!("Ada"),
    )
    .await;

    // The caret in light-DOM text slotted into an editor takes the text, open
    // or closed shadow root.
    for (label, host) in [
        ("Open slotted editor", "slotted-ce-open"),
        ("Closed slotted editor", "slotted-ce-closed"),
    ] {
        let setup =
            json!({ "id": host, "action": "evaluate", "script": format!("caretAtEnd('{host}')") });
        assert_success(&execute_command(&setup, &mut state).await);
        let selector = ref_for(label);
        for (action, key, text) in [("fill", "value", "X"), ("type", "text", "Y")] {
            let resp = execute_command(
                &json!({ "id": action, "action": action, "selector": selector, key: text }),
                &mut state,
            )
            .await;
            assert_success(&resp);
        }
        let script = format!("document.getElementById('{host}').textContent");
        assert_evaluate(&mut state, host, &script, json!("helloXY")).await;
    }

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Where `settle_focus` found focus: the id of the element inside the host,
/// or why not.
async fn settled_on(
    client: &super::cdp::client::CdpClient,
    session: &str,
    settled: super::interaction::Settled,
) -> String {
    use super::interaction::Settled;
    match settled {
        Settled::Inside(object_id) => {
            let id = client
                .send_command(
                    "Runtime.callFunctionOn",
                    Some(json!({
                        "objectId": object_id,
                        "functionDeclaration": "function() { return this.id; }",
                        "returnByValue": true,
                    })),
                    Some(session),
                )
                .await
                .unwrap();
            format!("#{}", id["result"]["value"].as_str().unwrap_or_default())
        }
        Settled::PulledOut => "pulled out".to_string(),
        Settled::Kept => "not passed on".to_string(),
        Settled::Out => "passed out".to_string(),
        Settled::Lost => "lost".to_string(),
    }
}

#[tokio::test]
#[ignore]
async fn e2e_focus_forwarding_is_read_from_the_focus_call_alone() {
    use super::interaction::{
        focus_call, focus_host, focusable, read_focus, settle_focus, settle_read, FocusCall,
    };

    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = r#"<input id="other">
        <div id="box"><input id="box-input"></div>
        <div id="later" tabindex="0"><input id="later-input"></div>
        <a id="link" href="/next"><input id="link-input" value="KEEP"></a>
        <details open><summary id="summary"><input id="summary-input" value="KEEP"></summary></details>
        <div><div id="editor" contenteditable><input id="editor-input" value="KEEP"></div></div>
        <x-dual id="dual" tabindex="0"><input id="dual-slotted"></x-dual>
        <x-closed-fwd id="closed-fwd" tabindex="0"></x-closed-fwd>
        <script>
            document.getElementById('later').addEventListener('focus', () => {
                queueMicrotask(() => document.getElementById('later-input').focus());
            });
            customElements.define('x-dual', class extends HTMLElement {
                constructor() {
                    super();
                    const root = this.attachShadow({ mode: 'closed' });
                    root.innerHTML = '<input id="internal"><slot></slot>';
                    window.dualInternal = root.getElementById('internal');
                    // Focus moving into the closed root reaches the host as a
                    // focus event too, so the handler passes focus on once.
                    window.dualForwards = true;
                    this.addEventListener('focus', () => {
                        if (dualForwards) document.getElementById('dual-slotted').focus();
                        dualForwards = false;
                    });
                }
            });
            customElements.define('x-closed-fwd', class extends HTMLElement {
                constructor() {
                    super();
                    const root = this.attachShadow({ mode: 'closed' });
                    root.innerHTML = '<input id="closed-fwd-input">';
                    const input = root.getElementById('closed-fwd-input');
                    this.addEventListener('focus', () => input.focus());
                }
            });
        </script>"#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let browser = state.browser.as_ref().unwrap();
    let client = &browser.client;
    let session = browser.active_session_id().unwrap();
    let run = |script: String| async move {
        client
            .send_command(
                "Runtime.evaluate",
                Some(json!({ "expression": script })),
                Some(session),
            )
            .await
            .unwrap()
    };
    // Between the focus call and the read, as on a slow or remote browser:
    // a timer focuses the input inside a container that can't take focus, or
    // makes the container focusable while focus is already inside. Neither is
    // the container passing focus on. A wrapper that passes focus on in a
    // microtask still does, however long until the read.
    for (start, host, between, expected) in [
        (
            "other",
            "box",
            "document.getElementById('box-input').focus()",
            "not passed on",
        ),
        (
            "box-input",
            "box",
            "document.getElementById('box').tabIndex = 0",
            "not passed on",
        ),
        ("other", "later", "", "#later-input"),
    ] {
        run(format!(
            "document.getElementById('box').removeAttribute('tabindex'); document.getElementById('{start}').focus()"
        ))
        .await;
        let object = run(format!("document.getElementById('{host}')")).await;
        let object_id = object["result"]["objectId"].as_str().unwrap().to_string();
        let call = focus_host(client, session, &object_id, false)
            .await
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        run(between.to_string()).await;
        let settled = settle_focus(client, session, &object_id, call)
            .await
            .unwrap();
        assert_eq!(
            settled_on(client, session, settled).await,
            expected,
            "{host} from {start}, then {between}"
        );
    }

    // With focus on the input inside, each wrapper can take focus when that
    // is read, but loses what makes it focusable before the call that focuses
    // it (a tabindex, a link's href, being its details' first summary, being
    // an editing host rather than content inside one), so its focus() does
    // nothing. The stale answer is not the wrapper passing focus on, so fill
    // leaves KEEP alone.
    for (host, change) in [
        ("box", "el.removeAttribute('tabindex')"),
        ("link", "el.removeAttribute('href')"),
        ("summary", "el.before(document.createElement('summary'))"),
        ("editor", "el.parentElement.contentEditable = 'true'"),
    ] {
        run(format!(
            "document.getElementById('box').tabIndex = 0; document.querySelector('#{host} input').focus()"
        ))
        .await;
        let object = run(format!("document.getElementById('{host}')")).await;
        let object_id = object["result"]["objectId"].as_str().unwrap().to_string();
        let FocusCall::Inside(inputs) = focus_call(client, session, &object_id, false, None)
            .await
            .unwrap()
        else {
            panic!("{host}: focus inside should be read first");
        };
        assert!(focusable(client, session, &object_id).await, "{host}");
        let changed = run(format!(
            "(() => {{ const el = document.getElementById('{host}'); {change}; }})()"
        ))
        .await;
        assert!(
            changed.get("exceptionDetails").is_none(),
            "{host}: {changed}"
        );
        let FocusCall::Done(call) =
            focus_call(client, session, &object_id, false, Some((inputs, true)))
                .await
                .unwrap()
        else {
            panic!("{host}: the call with the answer should focus");
        };
        let settled = settle_focus(client, session, &object_id, call)
            .await
            .unwrap();
        assert_eq!(
            settled_on(client, session, settled).await,
            "not passed on",
            "{host}: {change}"
        );
    }

    // The host passes focus to its slotted input, and the first read finds it
    // there. Before the next read, page code moves it into the host's closed
    // shadow root, where it reads as on the host: that is followed to the
    // internal input. Moved out of the host altogether, it is lost, and
    // nothing is filled; never the label's control. So too when the first
    // read finds focus in the host's closed root and it leaves before that
    // root is read.
    let other = "document.getElementById('other').focus()";
    for (host, between, expected) in [
        ("dual", "", "#dual-slotted"),
        ("dual", "dualInternal.focus()", "#internal"),
        ("dual", other, "lost"),
        ("closed-fwd", "", "#closed-fwd-input"),
        ("closed-fwd", other, "lost"),
    ] {
        run(format!("dualForwards = true; {other}")).await;
        let object = run(format!("document.getElementById('{host}')")).await;
        let object_id = object["result"]["objectId"].as_str().unwrap().to_string();
        let call = focus_host(client, session, &object_id, false)
            .await
            .unwrap();
        let read = read_focus(client, session, call).await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        run(between.to_string()).await;
        let settled = settle_read(client, session, &object_id, read)
            .await
            .unwrap();
        assert_eq!(
            settled_on(client, session, settled).await,
            expected,
            "{host}, then {between}"
        );
    }

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_fill_and_type_reach_a_forwarding_wrapper_in_a_background_tab() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // A wrapper whose focus handler passes focus to the input inside it.
    let html = r#"<input id="other" value="Jane">
        <div id="wrap" role="textbox" tabindex="0" aria-label="Wrapper"><input id="inner"></div>
        <script>
            document.getElementById('wrap').addEventListener('focus', () => document.getElementById('inner').focus());
        </script>"#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    // Another tab comes to the front. Chrome then holds focus events on this
    // page until it gains focus again, so the wrapper's handler only runs in
    // time if the page is made to behave as focused. agent-browser switches
    // to a new tab, so switch back, then bring the other tab to the front
    // without telling it.
    let browser = state.browser.as_ref().unwrap();
    let created = browser
        .client
        .send_command(
            "Target.createTarget",
            Some(json!({ "url": "about:blank" })),
            None,
        )
        .await
        .unwrap();
    let resp = execute_command(
        &json!({ "id": "3", "action": "tab_switch", "tabId": "t1" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let browser = state.browser.as_ref().unwrap();
    browser
        .client
        .send_command(
            "Target.activateTarget",
            Some(json!({ "targetId": created["targetId"] })),
            None,
        )
        .await
        .unwrap();
    assert_evaluate(
        &mut state,
        "background",
        "[document.visibilityState, document.hasFocus()]",
        json!(["hidden", false]),
    )
    .await;
    let resp = execute_command(
        &json!({ "id": "4", "action": "snapshot", "interactive": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap_or_default();
    let wrapper = snapshot
        .lines()
        .find(|line| line.contains("\"Wrapper\""))
        .and_then(|line| line.split("[ref=").nth(1))
        .and_then(|rest| rest.split([']', ',']).next())
        .map(|id| format!("@{id}"))
        .unwrap_or_else(|| panic!("no ref for the wrapper in {snapshot}"));

    for (action, key, text) in [("fill", "value", "X"), ("type", "text", "Y")] {
        let resp = execute_command(
            &json!({ "id": action, "action": action, "selector": wrapper, key: text }),
            &mut state,
        )
        .await;
        assert_success(&resp);
    }
    assert_evaluate(
        &mut state,
        "inner",
        "[document.getElementById('inner').value, document.getElementById('other').value]",
        json!(["XY", "Jane"]),
    )
    .await;

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_fill_and_type_reach_a_forwarding_wrapper_in_a_cross_origin_frame_in_a_background_tab()
{
    // A frame from localhost, another origin than the page on 127.0.0.1, holds
    // a wrapper whose focus handler passes focus to the input inside it. The
    // frame reports the input's value to the page by message.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let page = Arc::new(format!(
        r#"<iframe id="cross" name="cross" src="http://localhost:{port}/wrap"></iframe>
        <script>
            window.innerValue = '';
            addEventListener('message', (event) => {{ innerValue = event.data; }});
        </script>"#
    ));
    let frame = r#"<div id="wrap" role="textbox" tabindex="0"><input id="inner"></div><script>
        const inner = document.getElementById('inner');
        document.getElementById('wrap').addEventListener('focus', () => inner.focus());
        inner.addEventListener('input', () => parent.postMessage(inner.value, '*'));
    </script>"#;
    let server = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let page = page.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let body = if buf[..n].starts_with(b"GET /wrap") {
                    frame
                } else {
                    page.as_str()
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    });

    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let url = format!("http://127.0.0.1:{port}/");
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Another tab comes to the front, as in the background-tab test.
    let browser = state.browser.as_ref().unwrap();
    let created = browser
        .client
        .send_command(
            "Target.createTarget",
            Some(json!({ "url": "about:blank" })),
            None,
        )
        .await
        .unwrap();
    let resp = execute_command(
        &json!({ "id": "3", "action": "tab_switch", "tabId": "t1" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let browser = state.browser.as_ref().unwrap();
    browser
        .client
        .send_command(
            "Target.activateTarget",
            Some(json!({ "targetId": created["targetId"] })),
            None,
        )
        .await
        .unwrap();
    assert_evaluate(
        &mut state,
        "background",
        "[document.visibilityState, document.hasFocus()]",
        json!(["hidden", false]),
    )
    .await;

    let resp = execute_command(
        &json!({ "id": "4", "action": "snapshot", "interactive": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap_or_default();
    let frame_ref = snapshot
        .lines()
        .find(|line| line.contains("Iframe"))
        .and_then(|line| line.split("[ref=").nth(1))
        .and_then(|rest| rest.split([']', ',']).next())
        .map(|id| format!("@{id}"))
        .unwrap_or_else(|| panic!("no ref for the frame in {snapshot}"));
    let resp = execute_command(
        &json!({ "id": "4", "action": "frame", "selector": frame_ref }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    for (action, key, text) in [("fill", "value", "X"), ("type", "text", "Y")] {
        let resp = execute_command(
            &json!({ "id": action, "action": action, "selector": "#wrap", key: text }),
            &mut state,
        )
        .await;
        assert_success(&resp);
    }
    let resp = execute_command(&json!({ "id": "5", "action": "mainframe" }), &mut state).await;
    assert_success(&resp);
    // The message may arrive after the command returns. The page may be
    // hidden, so poll from here rather than with an in-page wait.
    let mut inner = Value::Null;
    for _ in 0..50 {
        let resp = execute_command(
            &json!({ "id": "inner", "action": "evaluate", "script": "innerValue" }),
            &mut state,
        )
        .await;
        inner = get_data(&resp)["result"].clone();
        if inner == json!("XY") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert_eq!(inner, json!("XY"));

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_fill_and_type_reach_a_closed_shadow_input_in_a_frame() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // After `frame`, the host is resolved through the parent document while
    // its closed shadow root resolves in the frame's own context.
    let html = r#"<iframe id="closed-frame" srcdoc="<div id='host'></div><script>
        const root = document.getElementById('host').attachShadow({ mode: 'closed', delegatesFocus: true });
        root.innerHTML = '<input>';
        window.hostInput = root.querySelector('input');
    </script>"></iframe>"#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    for (action, key, text, expected) in [
        ("fill", "value", "Ada", "Ada"),
        ("type", "text", "!", "Ada!"),
    ] {
        let steps = [
            json!({ "action": "frame", "selector": "#closed-frame" }),
            json!({ "action": action, "selector": "#host", key: text }),
            json!({ "action": "mainframe" }),
        ];
        for mut cmd in steps {
            cmd["id"] = json!(action);
            let resp = execute_command(&cmd, &mut state).await;
            assert_success(&resp);
        }
        assert_evaluate(
            &mut state,
            action,
            "document.getElementById('closed-frame').contentWindow.hostInput.value",
            json!(expected),
        )
        .await;
    }

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_fill_and_type_stay_in_an_iframe_editor() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // A TinyMCE-style editor: a same-origin frame whose body is the editing
    // host (srcdoc keeps the frame same-origin with the data: page).
    let html = r#"<input id="top-field">
        <iframe id="editor" title="Editor" srcdoc="<body contenteditable><p id='para'>hello</p></body>"></iframe>
        <iframe id="inert-editor" srcdoc="<body contenteditable inert><p id='para'>hello</p></body>"></iframe>"#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "3", "action": "fill", "selector": "#top-field", "value": "Jane" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // The <p> can't take focus and its frame doesn't have it, so the text
    // used to land in #top-field, which still held focus (#2055).
    for (action, key, text) in [("fill", "value", "[F]"), ("type", "text", "[T]")] {
        let steps = [
            json!({ "action": "evaluate", "script": "document.getElementById('top-field').focus()" }),
            json!({ "action": "frame", "selector": "#editor" }),
            json!({ "action": action, "selector": "#para", key: text }),
            json!({ "action": "mainframe" }),
        ];
        for mut cmd in steps {
            cmd["id"] = json!(action);
            let resp = execute_command(&cmd, &mut state).await;
            assert_success(&resp);
        }
        let script = format!(
            "[document.getElementById('top-field').value,
              document.getElementById('editor').contentDocument.body.textContent.includes('{text}')]"
        );
        assert_evaluate(&mut state, action, &script, json!(["Jane", true])).await;
    }

    // The frame itself, by ref, stands for its editing host, the second time
    // with focus already inside it.
    let resp = execute_command(
        &json!({ "id": "4", "action": "snapshot", "interactive": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap_or_default();
    let frame_ref = snapshot
        .lines()
        .find(|line| line.contains("Iframe \"Editor\""))
        .and_then(|line| line.split("[ref=").nth(1))
        .and_then(|rest| rest.split([']', ',']).next())
        .map(|id| format!("@{id}"))
        .unwrap_or_else(|| panic!("no ref for the editor frame in {snapshot}"));
    let focus_top = "document.getElementById('top-field').focus()";
    let setup = json!({ "id": "ref", "action": "evaluate", "script": focus_top });
    assert_success(&execute_command(&setup, &mut state).await);
    for text in ["[R1]", "[R2]"] {
        let resp = execute_command(
            &json!({ "id": "ref", "action": "fill", "selector": frame_ref, "value": text }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        let script = format!(
            "[document.getElementById('top-field').value,
              document.getElementById('editor').contentDocument.body.textContent.includes('{text}')]"
        );
        assert_evaluate(&mut state, "ref", &script, json!(["Jane", true])).await;
    }

    // An editing host that can't take focus leaves its frame unfocused while
    // the frame's body still reads as its activeElement; that must fail.
    let steps = [
        json!({ "action": "evaluate", "script": "document.getElementById('top-field').focus()" }),
        json!({ "action": "frame", "selector": "#inert-editor" }),
        json!({ "action": "fill", "selector": "#para", "value": "[I]" }),
        json!({ "action": "mainframe" }),
    ];
    let mut results = Vec::new();
    for mut cmd in steps {
        cmd["id"] = json!("inert");
        results.push(execute_command(&cmd, &mut state).await);
    }
    assert_eq!(results[2]["success"], false, "{}", results[2]);
    assert!(
        results[2]["error"]
            .as_str()
            .unwrap_or_default()
            .contains("did not take focus"),
        "{}",
        results[2]
    );
    assert_evaluate(
        &mut state,
        "inert",
        "document.getElementById('top-field').value",
        json!("Jane"),
    )
    .await;

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_fill_and_type_reach_an_editor_in_a_cross_origin_frame() {
    // The page on 127.0.0.1 frames an editor served from localhost, another
    // origin, so page JS can't read the frame's document. The editor holds
    // focus inside its frame and reports its text to the page by message. The
    // frame sits in a label whose control is another input.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let page = Arc::new(format!(
        r#"<input id="top-field">
        <label><iframe id="cross" src="http://localhost:{port}/editor"></iframe> <input id="labelled" value="KEEP"></label>
        <iframe id="same" srcdoc="<p>text</p>"></iframe>
        <script>
            window.editorText = '';
            addEventListener('message', (event) => {{ editorText = event.data; }});
        </script>"#
    ));
    let editor = r#"<div id="editor" contenteditable>hello</div><script>
        const editor = document.getElementById('editor');
        editor.focus();
        getSelection().collapse(editor, editor.childNodes.length);
        editor.addEventListener('input', () => parent.postMessage(editor.textContent, '*'));
    </script>"#;
    let server = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let page = page.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let body = if buf[..n].starts_with(b"GET /editor") {
                    editor
                } else {
                    page.as_str()
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    });

    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let url = format!("http://127.0.0.1:{port}/");
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "3", "action": "fill", "selector": "#top-field", "value": "Jane" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Text sent to the frame goes to whatever has focus inside it; fill
    // appends there, as on other editable content. The frame stands for
    // itself, not for its label's control.
    for (action, key, text, expected) in [
        ("fill", "value", "X", "helloX"),
        ("type", "text", "Y", "helloXY"),
    ] {
        let resp = execute_command(
            &json!({ "id": action, "action": action, "selector": "#cross", key: text }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert_evaluate(
            &mut state,
            action,
            "document.getElementById('labelled').value",
            json!("KEEP"),
        )
        .await;
        let check = format!("editorText === '{expected}'");
        let wait = json!({ "id": "wait", "action": "wait", "function": check, "timeout": 5000 });
        assert_success(&execute_command(&wait, &mut state).await);
    }

    // A frame the page can read is still refused when nothing focused inside
    // it takes text.
    let resp = execute_command(
        &json!({ "id": "same", "action": "fill", "selector": "#same", "value": "Z" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false, "{resp}");
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or_default()
            .contains("is not an <input>, <textarea> or [contenteditable] element"),
        "{resp}"
    );
    assert_evaluate(
        &mut state,
        "top",
        "document.getElementById('top-field').value",
        json!("Jane"),
    )
    .await;

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_fill_returns_on_pages_with_elements_named_host() {
    // A document's named properties shadow `host`: with a form or frame named
    // "host", walking up from the focused field past the document must not
    // follow it. Pages come from 127.0.0.1; the cross-origin frame from
    // localhost. The same-origin frame's own element with id "host" is what
    // its window's `host` names.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request.split_whitespace().nth(1).unwrap_or("/");
                let named = match path {
                    "/form" => r#"<form name="host"></form>"#.to_string(),
                    "/frame" => r#"<iframe name="host" srcdoc="<p id='host'>frame</p>"></iframe>"#
                        .to_string(),
                    "/cross" => {
                        format!(
                            r#"<iframe name="host" src="http://localhost:{port}/inner"></iframe>"#
                        )
                    }
                    _ => String::new(),
                };
                let body = format!(
                    r#"<input id="name"><input id="hidden" style="display:none"><input id="plain">{named}"#
                );
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    });

    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    for page in ["form", "frame", "cross"] {
        let url = format!("http://127.0.0.1:{port}/{page}");
        let resp = execute_command(
            &json!({ "id": page, "action": "navigate", "url": url }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        let resp = execute_command(
            &json!({ "id": page, "action": "fill", "selector": "#name", "value": "Jane" }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        // With focus on #name, refusing the hidden input walks up from #name
        // past the document; it must come back promptly. The normal input
        // still fills.
        for (selector, value, refused) in [("#hidden", "leak", true), ("#plain", "Ada", false)] {
            let cmd = json!({ "id": page, "action": "fill", "selector": selector, "value": value });
            let resp = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                execute_command(&cmd, &mut state),
            )
            .await
            .unwrap_or_else(|_| panic!("{page}: fill {selector} did not return"));
            if refused {
                assert_eq!(resp["success"], false, "{page}: {resp}");
                assert!(
                    resp["error"]
                        .as_str()
                        .unwrap_or_default()
                        .contains("did not take focus"),
                    "{page}: {resp}"
                );
            } else {
                assert_success(&resp);
            }
        }
        assert_evaluate(
            &mut state,
            page,
            "[document.getElementById('name').value, document.getElementById('plain').value]",
            json!(["Jane", "Ada"]),
        )
        .await;
    }

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_fill_returns_in_forms_with_controls_named_after_node_properties() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // A form's named controls shadow its own properties: here parentNode and
    // assignedSlot lead from the form to a control inside it and back, and
    // in an editor, parentElement does. Walking up from focus inside the form
    // must still end: to a field beside it, to a wrapper around it, and into
    // a textbox whose slot has nodes assigned. The editor's span hands its
    // text to the editing host, not to the control named parentElement.
    let html = r#"<form><input id="beside"><input id="field">
        <div id="wrap" tabindex="0"><input id="wrap-input"></div>
        <input name="parentNode"><input name="assignedSlot"></form>
        <x-named-box><span>slotted</span></x-named-box>
        <div id="editor" contenteditable><form><span id="span">text</span><input id="named" name="parentElement"></form></div>
        <script>
            customElements.define('x-named-box', class extends HTMLElement {
                constructor() {
                    super();
                    this.attachShadow({ mode: 'open' }).innerHTML =
                        '<div role="textbox" tabindex="0" aria-label="Named form box"><slot></slot></div>';
                }
            });
        </script>"#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "3", "action": "snapshot", "interactive": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap_or_default();
    let named_box = snapshot
        .lines()
        .find(|line| line.contains("\"Named form box\""))
        .and_then(|line| line.split("[ref=").nth(1))
        .and_then(|rest| rest.split([']', ',']).next())
        .map(|id| format!("@{id}"))
        .unwrap_or_else(|| panic!("no ref for the named form box in {snapshot}"));

    let not_text = "is not an <input>, <textarea> or [contenteditable] element";
    let focus_beside = "document.getElementById('beside').focus()";
    for (setup, action, selector, refused) in [
        (focus_beside, "fill", "#field".to_string(), None),
        (focus_beside, "type", "#field".to_string(), None),
        (
            "document.getElementById('wrap-input').focus()",
            "fill",
            "#wrap".to_string(),
            Some(not_text),
        ),
        (focus_beside, "fill", named_box, Some(not_text)),
        (
            "document.activeElement.blur()",
            "fill",
            "#span".to_string(),
            None,
        ),
    ] {
        let setup = json!({ "id": "setup", "action": "evaluate", "script": setup });
        assert_success(&execute_command(&setup, &mut state).await);
        let key = if action == "fill" { "value" } else { "text" };
        let cmd = json!({ "id": action, "action": action, "selector": selector, key: "X" });
        let resp = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            execute_command(&cmd, &mut state),
        )
        .await
        .unwrap_or_else(|_| panic!("{action} {selector} did not return"));
        match refused {
            Some(error) => {
                assert_eq!(resp["success"], false, "{action} {selector}: {resp}");
                assert!(
                    resp["error"].as_str().unwrap_or_default().contains(error),
                    "{action} {selector}: {resp}"
                );
            }
            None => assert_success(&resp),
        }
    }
    assert_evaluate(
        &mut state,
        "values",
        "[document.getElementById('field').value, document.getElementById('editor').textContent.includes('X'), document.getElementById('named').value]",
        json!(["XX", true, ""]),
    )
    .await;

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_select_option_label_override_names() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<html><body></body></html>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let script = r#"(() => {
        const select = document.createElement('select');
        select.id = 'label-overrides';
        for (const [value, label, text] of [
            ['initial', 'Initial', 'Initial'],
            ['target', 'Capital\u00A0Federal', 'Target source'],
            ['decoy', 'Other Province', 'Capital\u00A0\u00A0Federal'],
            ['hidden', 'Visible Province', 'Hidden\u00A0Only'],
            ['plain', null, 'Plain\u00A0Text']
        ]) {
            const option = document.createElement('option');
            option.value = value;
            if (label !== null) option.label = label;
            option.textContent = text;
            select.appendChild(option);
        }
        select.value = 'initial';
        select.dataset.changes = '0';
        select.addEventListener('change', () => select.dataset.changes++);
        document.body.appendChild(select);
        const multi = select.cloneNode(true);
        multi.id = 'label-overrides-multi';
        multi.multiple = true;
        multi.value = 'initial';
        multi.addEventListener('change', () => multi.dataset.changes++);
        document.body.appendChild(multi);
    })()"#;
    let resp = execute_command(
        &json!({ "id": "3", "action": "evaluate", "script": script }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let snapshot = execute_command(&json!({ "id": "4", "action": "snapshot" }), &mut state).await;
    let mut results = Vec::new();
    for (selector, values, expected_values, changes, success) in [
        (
            "#label-overrides",
            vec!["Capital Federal"],
            vec!["target"],
            "1",
            true,
        ),
        (
            "#label-overrides",
            vec!["Hidden Only"],
            vec!["target"],
            "1",
            false,
        ),
        ("#label-overrides", vec!["decoy"], vec!["decoy"], "2", true),
        (
            "#label-overrides",
            vec!["Capital\u{00A0}Federal"],
            vec!["target"],
            "3",
            true,
        ),
        (
            "#label-overrides",
            vec!["Hidden\u{00A0}Only"],
            vec!["hidden"],
            "4",
            true,
        ),
        (
            "#label-overrides",
            vec!["Plain Text"],
            vec!["plain"],
            "5",
            true,
        ),
        (
            "#label-overrides-multi",
            vec!["target", "Hidden Only"],
            vec!["initial"],
            "0",
            false,
        ),
    ] {
        let selection = select_values(&mut state, "select", selector, &values).await;
        let script = format!(
            "(() => {{ const select = document.querySelector('{}'); return {{ values: [...select.selectedOptions].map(option => option.value), changes: select.dataset.changes }}; }})()",
            selector
        );
        let actual = execute_command(
            &json!({ "id": "state", "action": "evaluate", "script": script }),
            &mut state,
        )
        .await;
        results.push((selection, actual, expected_values, changes, success));
    }
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    assert_success(&snapshot);
    let snapshot = get_data(&snapshot)["snapshot"].as_str().unwrap();
    assert!(snapshot.contains("option \"Capital Federal\""));
    assert!(snapshot.contains("option \"Other Province\""));
    assert!(!snapshot.contains("option \"Hidden Only\""));
    for (selection, actual, expected_values, changes, success) in results {
        assert_eq!(selection["success"], success, "{selection}");
        if !success {
            assert!(selection["error"]
                .as_str()
                .unwrap()
                .contains("No option matched"));
        }
        assert_success(&actual);
        assert_eq!(
            get_data(&actual)["result"],
            json!({ "values": expected_values, "changes": changes })
        );
    }
}

#[tokio::test]
#[ignore]
async fn e2e_select_option_normalized_names() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<html><body></body></html>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let script = r#"(() => {
        const add = (select, value, label, selected = false) => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = label;
            option.selected = selected;
            select.appendChild(option);
        };
        const province = document.createElement('select');
        province.id = 'province';
        add(province, '', 'Provincia');
        add(province, 'B', 'Buenos\u00A0Aires', true);
        add(province, 'C', 'Capital\u00A0Federal');
        add(province, 'Z', 'Zero\u200BWidth');
        province.dataset.changes = '0';
        province.addEventListener('change', () => province.dataset.changes++);
        document.body.appendChild(province);

        const collision = document.createElement('select');
        collision.id = 'collision';
        add(collision, 'ascii', 'Alpha Beta', true);
        add(collision, 'nbsp', 'Alpha\u00A0Beta');
        document.body.appendChild(collision);

        const cases = document.createElement('select');
        cases.id = 'cases';
        add(cases, 'US', 'Upper');
        add(cases, 'us', 'Lower');
        document.body.appendChild(cases);

        const multi = document.createElement('select');
        multi.id = 'multi';
        multi.multiple = true;
        add(multi, 'a', 'One\u00A0A');
        add(multi, 'b', 'Two B');
        document.body.appendChild(multi);
    })()"#;
    let resp = execute_command(
        &json!({ "id": "3", "action": "evaluate", "script": script }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "4", "action": "snapshot" }), &mut state).await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap();
    assert!(snapshot.contains("option \"Capital Federal\""));
    assert!(!snapshot.contains("CapitalFederal"));
    assert!(snapshot.contains("option \"ZeroWidth\""));

    let resp = select_values(&mut state, "5", "#province", &["Capital Federal"]).await;
    assert_success(&resp);
    assert_evaluate(
        &mut state,
        "6",
        "({ value: province.value, changes: province.dataset.changes })",
        json!({ "value": "C", "changes": "1" }),
    )
    .await;

    let resp = select_values(&mut state, "7", "#province", &["missing"]).await;
    assert_eq!(resp["success"], false);
    assert_evaluate(
        &mut state,
        "8",
        "({ value: province.value, changes: province.dataset.changes })",
        json!({ "value": "C", "changes": "1" }),
    )
    .await;

    let resp = select_values(&mut state, "9", "#collision", &["Alpha Beta"]).await;
    assert_success(&resp);
    assert_evaluate(&mut state, "10", "collision.value", json!("ascii")).await;

    let resp = select_values(&mut state, "11", "#collision", &["Alpha  Beta"]).await;
    assert_eq!(resp["success"], false);
    assert!(resp["error"]
        .as_str()
        .unwrap()
        .contains("Multiple options matched"));
    assert_evaluate(&mut state, "12", "collision.value", json!("ascii")).await;

    let resp = select_values(&mut state, "13", "#cases", &["us"]).await;
    assert_success(&resp);
    assert_evaluate(&mut state, "14", "cases.value", json!("us")).await;

    let resp = select_values(&mut state, "15", "#multi", &["One A", "b"]).await;
    assert_success(&resp);
    assert_evaluate(
        &mut state,
        "16",
        "[...multi.selectedOptions].map(option => option.value)",
        json!(["a", "b"]),
    )
    .await;

    let resp = select_values(&mut state, "17", "#multi", &["a", "missing"]).await;
    assert_eq!(resp["success"], false);
    assert_evaluate(
        &mut state,
        "18",
        "[...multi.selectedOptions].map(option => option.value)",
        json!(["a", "b"]),
    )
    .await;

    let resp = select_values(&mut state, "19", "#province", &["ZeroWidth"]).await;
    assert_success(&resp);
    assert_evaluate(&mut state, "20", "province.value", json!("Z")).await;

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Navigation: back, forward, reload
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_navigation_history() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate to page 1
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<h1>Page 1</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate to page 2
    let resp = execute_command(
        &json!({ "id": "3", "action": "navigate", "url": "data:text/html,<h1>Page 2</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Back
    let resp = execute_command(&json!({ "id": "4", "action": "back" }), &mut state).await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "5", "action": "evaluate", "script": "document.querySelector('h1').textContent" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "Page 1");

    // Forward
    let resp = execute_command(&json!({ "id": "6", "action": "forward" }), &mut state).await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "7", "action": "evaluate", "script": "document.querySelector('h1').textContent" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "Page 2");

    // Reload
    let resp = execute_command(&json!({ "id": "8", "action": "reload" }), &mut state).await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "9", "action": "evaluate", "script": "document.querySelector('h1').textContent" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "Page 2");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Cookies
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_cookies() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Set cookie
    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "cookies_set",
            "name": "test_cookie",
            "value": "hello123"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Get cookies
    let resp = execute_command(&json!({ "id": "4", "action": "cookies_get" }), &mut state).await;
    assert_success(&resp);
    let cookies = get_data(&resp)["cookies"].as_array().unwrap();
    let found = cookies
        .iter()
        .any(|c| c["name"] == "test_cookie" && c["value"] == "hello123");
    assert!(found, "Should find the set cookie");

    // Clear cookies
    let resp = execute_command(&json!({ "id": "5", "action": "cookies_clear" }), &mut state).await;
    assert_success(&resp);

    // Verify cleared
    let resp = execute_command(&json!({ "id": "6", "action": "cookies_get" }), &mut state).await;
    assert_success(&resp);
    let cookies = get_data(&resp)["cookies"].as_array().unwrap();
    let found = cookies.iter().any(|c| c["name"] == "test_cookie");
    assert!(!found, "Cookie should be cleared");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// localStorage / sessionStorage
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_storage() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Set local storage
    let resp = execute_command(
        &json!({ "id": "3", "action": "storage_set", "type": "local", "key": "mykey", "value": "myvalue" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Get local storage key
    let resp = execute_command(
        &json!({ "id": "4", "action": "storage_get", "type": "local", "key": "mykey" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["value"], "myvalue");

    // Get all local storage
    let resp = execute_command(
        &json!({ "id": "5", "action": "storage_get", "type": "local" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["data"]["mykey"], "myvalue");

    // Clear
    let resp = execute_command(
        &json!({ "id": "6", "action": "storage_clear", "type": "local" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Verify cleared
    let resp = execute_command(
        &json!({ "id": "7", "action": "storage_get", "type": "local" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let data = &get_data(&resp)["data"];
    assert!(
        data.as_object().map(|m| m.is_empty()).unwrap_or(true),
        "Storage should be empty after clear"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Tab management
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_tabs() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<h1>Tab 1</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Tab list should show 1 tab with tabId 1
    let resp = execute_command(&json!({ "id": "3", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap();
    assert_eq!(tabs.len(), 1);
    assert_eq!(tabs[0]["active"], true);
    assert_eq!(tabs[0]["tabId"], "t1", "First tab should have tabId t1");

    // Open new tab
    let resp = execute_command(
        &json!({ "id": "4", "action": "tab_new", "url": "data:text/html,<h1>Tab 2</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["tabId"],
        "t2",
        "New tab should have tabId t2"
    );
    assert_eq!(get_data(&resp)["total"], 2);

    // Tab list should show 2 tabs with distinct, incrementing tabIds
    let resp = execute_command(&json!({ "id": "5", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap();
    assert_eq!(tabs.len(), 2);
    assert_eq!(tabs[1]["active"], true);
    assert_eq!(tabs[0]["tabId"], "t1", "First tab should keep tabId t1");
    assert_eq!(tabs[1]["tabId"], "t2", "Second tab should have tabId t2");

    // Switch to first tab
    let resp = execute_command(
        &json!({ "id": "6", "action": "tab_switch", "tabId": "t1" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "7", "action": "evaluate", "script": "document.querySelector('h1').textContent" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "Tab 1");

    // Close second tab
    let resp = execute_command(
        &json!({ "id": "8", "action": "tab_close", "tabId": "t2" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Should have 1 tab left
    let resp = execute_command(&json!({ "id": "9", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap();
    assert_eq!(tabs.len(), 1);

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_tab_ids_not_reused() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // First tab gets tabId 1
    let resp = execute_command(&json!({ "id": "2", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap();
    assert_eq!(tabs[0]["tabId"], "t1");

    // Open tab 2 and tab 3
    let resp = execute_command(
        &json!({ "id": "3", "action": "tab_new", "url": "data:text/html,<h1>Tab 2</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["tabId"], "t2");

    let resp = execute_command(
        &json!({ "id": "4", "action": "tab_new", "url": "data:text/html,<h1>Tab 3</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["tabId"], "t3");

    // Close tab 2
    let resp = execute_command(
        &json!({ "id": "5", "action": "tab_close", "tabId": "t2" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Open a new tab — should get tabId 4, NOT 2
    let resp = execute_command(
        &json!({ "id": "6", "action": "tab_new", "url": "data:text/html,<h1>Tab 4</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["tabId"],
        "t4",
        "Tab IDs must not be reused after closing"
    );

    // Verify final state: tabs t1, t3, t4
    let resp = execute_command(&json!({ "id": "7", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap();
    assert_eq!(tabs.len(), 3);
    let ids: Vec<String> = tabs
        .iter()
        .map(|t| t["tabId"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids, vec!["t1", "t3", "t4"]);

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// `tab_close` with an explicit `tabId` must close that tab regardless of
/// whether it's active, and leave the remaining tab active without leaking
/// per-tab state (refs, iframe sessions, frame id) from the closed tab.
#[tokio::test]
#[ignore]
async fn e2e_tab_close_with_tab_id_closes_active_tab() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<title>A</title>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "tab_new", "url": "data:text/html,<title>B</title>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "4", "action": "tab_close", "tabId": "t2" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "5", "action": "title" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["title"], "A");
    assert!(state.ref_map.get("e1").is_none());
    assert!(state.iframe_sessions.is_empty());
    assert!(state.active_frame_id.is_none());

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Tabs can be opened with a user-assigned label and then addressed by that
/// label anywhere a `t<N>` id is accepted (switch, close, and JSON `tabId`
/// on `tab_switch` / `tab_close`). Labels are the agent-friendly way to
/// write multi-tab workflows without memorizing ids.
#[tokio::test]
#[ignore]
async fn e2e_tab_new_with_label_can_be_switched_and_closed() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<title>Home</title>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Open a labeled tab and verify the response echoes the label and a
    // `t<N>` style tabId.
    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "tab_new",
            "url": "data:text/html,<title>Docs</title>",
            "label": "docs",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["tabId"], "t2");
    assert_eq!(get_data(&resp)["label"], "docs");

    // tab_list exposes the label alongside the id.
    let resp = execute_command(&json!({ "id": "4", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap();
    let docs = tabs
        .iter()
        .find(|t| t["tabId"] == "t2")
        .expect("docs tab should be present");
    assert_eq!(docs["label"], "docs");

    // tab_switch accepts the label.
    let resp = execute_command(
        &json!({ "id": "5", "action": "tab_switch", "tabId": "t1" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(state.browser.as_ref().unwrap().active_tab_id(), Some(1));

    let resp = execute_command(
        &json!({ "id": "6", "action": "tab_switch", "tabId": "docs" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(state.browser.as_ref().unwrap().active_tab_id(), Some(2));

    // Once switched, the active tab is the labeled one and normal commands
    // work against it.
    let resp = execute_command(&json!({ "id": "7", "action": "title" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["title"], "Docs");

    // tab_close accepts the label.
    let resp = execute_command(
        &json!({ "id": "8", "action": "tab_close", "tabId": "docs" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["label"], "docs");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Duplicate labels must be rejected so agents can treat a label as a unique
/// handle. The first tab keeps the label; the second tab's creation errors.
#[tokio::test]
#[ignore]
async fn e2e_tab_new_with_duplicate_label_errors() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "tab_new", "url": "about:blank", "label": "docs" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "tab_new", "url": "about:blank", "label": "docs" }),
        &mut state,
    )
    .await;
    assert_eq!(
        resp.get("success").and_then(|v| v.as_bool()),
        Some(false),
        "duplicate label should error: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );
    let err = resp.get("error").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        err.contains("already used"),
        "error should explain the collision: {}",
        err
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Positional integers passed as `tabId` on tab-switch / tab-close must be
/// rejected by the daemon-layer parser, not silently coerced. The error
/// should teach the user the correct form (`t<N>`).
#[tokio::test]
#[ignore]
async fn e2e_tab_switch_rejects_bare_integer() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "tab_switch", "tabId": "2" }),
        &mut state,
    )
    .await;
    assert_eq!(
        resp.get("success").and_then(|v| v.as_bool()),
        Some(false),
        "bare integer tabId on tab_switch should error: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );
    let err = resp.get("error").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        err.contains("t2") && err.contains("positional integers"),
        "error should teach `t<N>` convention: {}",
        err
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Element queries: isvisible, isenabled, gettext, getattribute
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_element_queries() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = concat!(
        "data:text/html,<html><body>",
        "<p id='visible'>Hello World</p>",
        "<p id='hidden' style='display:none'>Hidden</p>",
        "<input id='enabled' value='test'>",
        "<input id='disabled' disabled value='nope'>",
        "<a id='link' href='https://example.com' data-testid='my-link'>Click me</a>",
        "</body></html>"
    );

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // isvisible
    let resp = execute_command(
        &json!({ "id": "3", "action": "isvisible", "selector": "#visible" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["visible"], true);

    let resp = execute_command(
        &json!({ "id": "4", "action": "isvisible", "selector": "#hidden" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["visible"], false);

    // isenabled
    let resp = execute_command(
        &json!({ "id": "5", "action": "isenabled", "selector": "#enabled" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["enabled"], true);

    let resp = execute_command(
        &json!({ "id": "6", "action": "isenabled", "selector": "#disabled" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["enabled"], false);

    // gettext
    let resp = execute_command(
        &json!({ "id": "7", "action": "gettext", "selector": "#visible" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["text"], "Hello World");

    // getattribute
    let resp = execute_command(
        &json!({ "id": "8", "action": "getattribute", "selector": "#link", "attribute": "href" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["value"], "https://example.com");

    let resp = execute_command(
        &json!({ "id": "9", "action": "getattribute", "selector": "#link", "attribute": "data-testid" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["value"], "my-link");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_getbyrole_uses_accessibility_tree_for_implicit_roles() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = concat!(
        "<html><head>",
        "<link rel='stylesheet' href='data:text/css,body{}'>",
        "</head><body>",
        "<h1 style='text-transform:uppercase'>Welcome</h1>",
        "<a id='services' href='#services' onclick='window.__clicked = \"services\"'>Services</a>",
        "<button id='submit'>Submit</button>",
        "</body></html>"
    );
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Implicit role from a plain HTML tag (<h1> -> heading), matched
    // case-insensitively against the accessible name despite the CSS
    // text-transform rendering it as "WELCOME".
    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "getbyrole",
            "role": "heading",
            "name": "welcome",
            "subaction": "text"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["text"], "WELCOME");

    // A real <a href> must win over the unrelated <link rel="stylesheet">
    // element, which is not exposed as an AX "link" node at all.
    let resp = execute_command(
        &json!({
            "id": "4",
            "action": "getbyrole",
            "role": "link",
            "name": "Services",
            "exact": true,
            "subaction": "click"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "5", "action": "evaluate", "script": "window.__clicked" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "services");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Wait command
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_wait() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = concat!(
        "data:text/html,<html><body>",
        "<div id='target' style='display:none'>Appeared!</div>",
        "<script>setTimeout(() => document.getElementById('target').style.display='block', 500)</script>",
        "</body></html>"
    );

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Wait for selector to become visible
    let resp = execute_command(
        &json!({ "id": "3", "action": "wait", "selector": "#target", "state": "visible", "timeout": 5000 }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Wait for text
    let resp = execute_command(
        &json!({ "id": "4", "action": "wait", "text": "Appeared!", "timeout": 5000 }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Timeout wait
    let start = std::time::Instant::now();
    let resp = execute_command(
        &json!({ "id": "5", "action": "wait", "timeout": 200 }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(
        start.elapsed().as_millis() >= 150,
        "Timeout wait should sleep at least 150ms"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// wait --load on a page that already finished loading must resolve
// immediately instead of waiting for a Page.loadEventFired that will never
// come (the common case after a click that triggers an SPA navigation).
#[tokio::test]
#[ignore]
async fn e2e_wait_load_state_resolves_immediately_when_already_loaded() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<h1>Loaded</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    for (id, load_state) in [("3", "load"), ("4", "domcontentloaded")] {
        let start = std::time::Instant::now();
        let resp = execute_command(
            &json!({ "id": id, "action": "waitforloadstate", "state": load_state, "timeout": 10000 }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert!(
            start.elapsed().as_millis() < 3000,
            "wait --load {} on an already-loaded page should resolve immediately, took {}ms",
            load_state,
            start.elapsed().as_millis()
        );
    }

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Same-document navigation regression test
// ---------------------------------------------------------------------------
//
// Chrome may perform a same-document navigation when it determines the target
// URL is the same document as the current page (ignoring fragment). This
// causes Page.loadEventFired to not fire, making wait_for_lifecycle
// hang forever waiting for an event that never comes.
//
// The fix checks loader_id in the Page.navigate response - if None,
// it's a same-document navigation and we skip waiting for lifecycle events.

#[tokio::test]
#[ignore]
async fn e2e_navigate_same_url_twice_should_not_hang() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate to about:blank first to start from a known state
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "about:blank" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Create a simple HTML page that changes its own URL via history.pushState
    // This simulates SPA routing behavior which triggers same-document navigation
    let base_page = "data:text/html,<html><body><script>
        // On first load, change URL via pushState without navigation
        history.pushState({}, '', '/#/home');
    </script><h1>Test</h1></body></html>";

    // Navigate to the page (first time)
    let resp = execute_command(
        &json!({ "id": "3", "action": "navigate", "url": base_page }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Verify URL changed due to pushState
    let resp = execute_command(&json!({ "id": "4", "action": "url" }), &mut state).await;
    assert_success(&resp);
    let url_after_push = get_data(&resp)["url"].as_str().unwrap();
    // URL should have changed to include /#/home due to pushState
    assert!(
        url_after_push.contains("/%23/home") || url_after_push.contains("/#/home"),
        "URL should have changed via pushState, got: {}",
        url_after_push
    );

    // Navigate to the SAME base URL again
    // Without fix: Chrome may do same-document nav, wait_for_lifecycle hangs
    // With fix: We detect loader_id is None and skip waiting
    let start = std::time::Instant::now();
    let resp = execute_command(
        &json!({ "id": "5", "action": "navigate", "url": base_page }),
        &mut state,
    )
    .await;
    let elapsed = start.elapsed().as_secs();

    // Should complete quickly (< 5 seconds) without hanging
    // Without fix, this times out after 25 seconds (default_timeout_ms)
    assert!(
        elapsed < 5,
        "Second navigation should not hang, but took {}s",
        elapsed
    );
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Viewport with deviceScaleFactor (retina)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_viewport_scale_factor() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "about:blank" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Default devicePixelRatio should be 1
    let resp = execute_command(
        &json!({ "id": "3", "action": "evaluate", "script": "window.devicePixelRatio" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let default_dpr = get_data(&resp)["result"].as_f64().unwrap();
    assert_eq!(default_dpr, 1.0, "Default devicePixelRatio should be 1");

    // Set viewport with 2x scale factor
    let resp = execute_command(
        &json!({ "id": "4", "action": "viewport", "width": 1920, "height": 1080, "deviceScaleFactor": 2.0 }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["width"], 1920);
    assert_eq!(get_data(&resp)["height"], 1080);
    assert_eq!(get_data(&resp)["deviceScaleFactor"], 2.0);

    // devicePixelRatio should now be 2
    let resp = execute_command(
        &json!({ "id": "5", "action": "evaluate", "script": "window.devicePixelRatio" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let new_dpr = get_data(&resp)["result"].as_f64().unwrap();
    assert_eq!(
        new_dpr, 2.0,
        "devicePixelRatio should be 2 after setting scale factor"
    );

    // CSS viewport width should still be 1920 (not 3840)
    let resp = execute_command(
        &json!({ "id": "6", "action": "evaluate", "script": "window.innerWidth" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let css_width = get_data(&resp)["result"].as_i64().unwrap();
    assert_eq!(css_width, 1920, "CSS width should remain 1920 at 2x scale");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Viewport and emulation
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_viewport_emulation() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<h1>Viewport</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Get initial width
    let resp = execute_command(
        &json!({ "id": "3", "action": "evaluate", "script": "window.innerWidth" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let initial_width = get_data(&resp)["result"].as_i64().unwrap();

    // Set viewport to a different size
    let resp = execute_command(
        &json!({ "id": "4", "action": "viewport", "width": 375, "height": 812, "deviceScaleFactor": 3.0, "mobile": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["width"], 375);
    assert_eq!(get_data(&resp)["height"], 812);
    assert_eq!(get_data(&resp)["mobile"], true);

    // Reload to apply viewport change
    let resp = execute_command(&json!({ "id": "5", "action": "reload" }), &mut state).await;
    assert_success(&resp);

    // Width should differ from default (setDeviceMetricsOverride applied)
    let resp = execute_command(
        &json!({ "id": "6", "action": "evaluate", "script": "window.innerWidth" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let new_width = get_data(&resp)["result"].as_i64().unwrap();
    assert!(
        new_width != initial_width || new_width == 375,
        "Viewport should change from {} after setDeviceMetricsOverride (got {})",
        initial_width,
        new_width
    );

    // Set user agent
    let resp = execute_command(
        &json!({ "id": "5", "action": "user_agent", "userAgent": "TestBot/1.0" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "6", "action": "evaluate", "script": "navigator.userAgent" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "TestBot/1.0");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Hover, scroll, press
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_hover_scroll_press() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = concat!(
        "data:text/html,<html><body style='height:3000px'>",
        "<button id='btn' onmouseover=\"this.textContent='hovered'\">Hover me</button>",
        "<input id='input' onkeydown=\"this.dataset.key=event.key\">",
        "</body></html>"
    );

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Hover
    let resp = execute_command(
        &json!({ "id": "3", "action": "hover", "selector": "#btn" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Scroll
    let resp = execute_command(
        &json!({ "id": "4", "action": "scroll", "y": 500 }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "5", "action": "evaluate", "script": "window.scrollY" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let scroll_y = get_data(&resp)["result"].as_f64().unwrap();
    assert!(scroll_y > 0.0, "Should have scrolled down");

    // Press key
    let resp = execute_command(
        &json!({ "id": "6", "action": "press", "key": "Enter" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["pressed"], "Enter");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Raw mouse regressions
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_mouse_down_move_up_preserves_drag_state() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": native_test_fixture_url("drag_probe")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "evaluate",
            "script": r#"(() => {
                const rect = document.getElementById('target').getBoundingClientRect();
                return {
                    left: Math.round(rect.left),
                    top: Math.round(rect.top),
                    x: Math.round(rect.left + rect.width / 2),
                    y: Math.round(rect.top + rect.height / 2)
                };
            })()"#
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let start = &get_data(&resp)["result"];
    let initial_left = start["left"]
        .as_i64()
        .expect("target left should be numeric");
    let initial_top = start["top"].as_i64().expect("target top should be numeric");
    let start_x = start["x"].as_i64().expect("target x should be numeric");
    let start_y = start["y"].as_i64().expect("target y should be numeric");
    let end_x = start_x + 80;
    let end_y = start_y + 60;

    let resp = execute_command(
        &json!({ "id": "4", "action": "mousemove", "x": start_x, "y": start_y }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "5", "action": "mousedown", "button": "left" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "6", "action": "mousemove", "x": end_x, "y": end_y }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "7", "action": "mouseup", "button": "left" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "8", "action": "evaluate", "script": "window.__dragProbe" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let probe = &get_data(&resp)["result"];
    assert_eq!(probe["finalLeft"].as_i64(), Some(initial_left + 80));
    assert_eq!(probe["finalTop"].as_i64(), Some(initial_top + 60));

    let events = probe["events"]
        .as_array()
        .expect("drag probe should expose events");
    assert!(
        events.iter().any(|event| {
            event["type"] == "mousedown"
                && event["x"].as_f64() == Some(start_x as f64)
                && event["y"].as_f64() == Some(start_y as f64)
                && event["buttons"].as_i64() == Some(1)
        }),
        "Expected a non-zero mousedown event in drag probe"
    );
    assert!(
        events.iter().any(|event| {
            event["type"] == "mousemove"
                && event["x"].as_f64() == Some(end_x as f64)
                && event["y"].as_f64() == Some(end_y as f64)
                && event["buttons"].as_i64() == Some(1)
        }),
        "Expected a drag mousemove with the button still pressed"
    );
    assert!(
        events.iter().any(|event| {
            event["type"] == "mouseup"
                && event["x"].as_f64() == Some(end_x as f64)
                && event["y"].as_f64() == Some(end_y as f64)
                && event["buttons"].as_i64() == Some(0)
        }),
        "Expected mouseup at the last drag position"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_mouse_drag_reaches_pointer_capture_target() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": native_test_fixture_url("pointer_capture_probe")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "evaluate",
            "script": r#"(() => {
                const rect = document.getElementById('handle').getBoundingClientRect();
                return {
                    x: Math.round(rect.left + rect.width / 2),
                    y: Math.round(rect.top + rect.height / 2)
                };
            })()"#
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let start = &get_data(&resp)["result"];
    let start_x = start["x"].as_i64().expect("handle x should be numeric");
    let start_y = start["y"].as_i64().expect("handle y should be numeric");
    let end_x = start_x + 80;
    let end_y = start_y + 60;

    let resp = execute_command(
        &json!({ "id": "4", "action": "mousemove", "x": start_x, "y": start_y }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "5", "action": "mousedown", "button": "left" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "6", "action": "mousemove", "x": end_x, "y": end_y }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "7", "action": "mouseup", "button": "left" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "8", "action": "evaluate", "script": "window.__pointerCaptureProbe" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let probe = &get_data(&resp)["result"];
    assert_eq!(probe["moved"].as_bool(), Some(true));

    let events = probe["events"]
        .as_array()
        .expect("pointer capture probe should expose events");
    assert!(
        events.iter().any(|event| {
            event["type"] == "pointermove"
                && event["phase"] == "drag"
                && event["hasCapture"].as_bool() == Some(true)
                && event["x"].as_f64() == Some(end_x as f64)
                && event["y"].as_f64() == Some(end_y as f64)
        }),
        "Expected pointermove with capture during the drag"
    );
    assert!(
        events.iter().any(|event| {
            event["type"] == "pointerup"
                && event["phase"] == "up"
                && event["hadCapture"].as_bool() == Some(true)
        }),
        "Expected pointerup to observe an active pointer capture"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_drag_action_sends_buttons_during_move() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": native_test_fixture_url("html5_drag_probe")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "drag",
            "source": "#source",
            "target": "#dest"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["dragged"].as_bool(), Some(true));

    let resp = execute_command(
        &json!({ "id": "4", "action": "evaluate", "script": "window.__html5DragProbe" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let probe = &get_data(&resp)["result"];
    let events = probe["events"]
        .as_array()
        .expect("html5 drag probe should expose events");

    // The mousemove events emitted while the button is held should carry
    // buttons == 1 so the browser recognises the gesture as a drag.
    assert!(
        events
            .iter()
            .any(|event| { event["type"] == "mousemove" && event["buttons"].as_i64() == Some(1) }),
        "Expected at least one mousemove with buttons == 1 during drag"
    );

    // dragstart must fire on the source element.
    assert!(
        events.iter().any(|event| event["type"] == "dragstart"),
        "Expected dragstart to fire on the source element"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// State save/load, state management
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_state_management() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Set some storage
    let resp = execute_command(
        &json!({ "id": "3", "action": "storage_set", "type": "local", "key": "persist_key", "value": "persist_val" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Save state
    let tmp_state = std::env::temp_dir()
        .join("agent-browser-e2e-state.json")
        .to_string_lossy()
        .to_string();
    let resp = execute_command(
        &json!({ "id": "4", "action": "state_save", "path": &tmp_state }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(std::path::Path::new(&tmp_state).exists());

    // State show
    let resp = execute_command(
        &json!({ "id": "5", "action": "state_show", "path": &tmp_state }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let state_data = get_data(&resp);
    assert!(state_data.get("state").is_some());

    // State list
    let resp = execute_command(&json!({ "id": "6", "action": "state_list" }), &mut state).await;
    assert_success(&resp);
    assert!(get_data(&resp)["files"].is_array());

    // Clean up
    let _ = std::fs::remove_file(&tmp_state);

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Cross-domain state save (issue #1060)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_save_state_cross_domain() {
    let mut state = DaemonState::new();

    // Launch
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate to domain A and set cookie + localStorage
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://httpbin.org/html" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "3", "action": "cookies_set",
            "name": "domainA_cookie", "value": "from_httpbin"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "4", "action": "storage_set",
            "type": "local", "key": "domainA_key", "value": "domainA_val"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate to domain B and set cookie + localStorage
    let resp = execute_command(
        &json!({ "id": "5", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "6", "action": "cookies_set",
            "name": "domainB_cookie", "value": "from_example"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "7", "action": "storage_set",
            "type": "local", "key": "domainB_key", "value": "domainB_val"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Save state (currently on example.com)
    let tmp_state = std::env::temp_dir()
        .join("agent-browser-e2e-cross-domain-state.json")
        .to_string_lossy()
        .to_string();
    let resp = execute_command(
        &json!({ "id": "8", "action": "state_save", "path": &tmp_state }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Read and verify saved state
    let saved = std::fs::read_to_string(&tmp_state).expect("State file should exist");
    let state_data: serde_json::Value = serde_json::from_str(&saved).unwrap();

    // Verify BOTH domain cookies are present
    let cookies = state_data["cookies"].as_array().unwrap();
    let has_domain_a = cookies.iter().any(|c| c["name"] == "domainA_cookie");
    let has_domain_b = cookies.iter().any(|c| c["name"] == "domainB_cookie");
    assert!(
        has_domain_a,
        "Should include cross-domain cookie from httpbin.org: {:?}",
        cookies
    );
    assert!(
        has_domain_b,
        "Should include cookie from example.com: {:?}",
        cookies
    );

    // Verify BOTH origins' localStorage are present
    let origins = state_data["origins"].as_array().unwrap();
    let has_origin_a = origins.iter().any(|o| {
        o["origin"].as_str().is_some_and(|s| s.contains("httpbin"))
            && o["localStorage"]
                .as_array()
                .is_some_and(|ls| ls.iter().any(|e| e["name"] == "domainA_key"))
    });
    let has_origin_b = origins.iter().any(|o| {
        o["origin"].as_str().is_some_and(|s| s.contains("example"))
            && o["localStorage"]
                .as_array()
                .is_some_and(|ls| ls.iter().any(|e| e["name"] == "domainB_key"))
    });
    assert!(
        has_origin_a,
        "Should include localStorage from httpbin.org origin: {:?}",
        origins
    );
    assert!(
        has_origin_b,
        "Should include localStorage from example.com origin: {:?}",
        origins
    );

    // Clean up
    let _ = std::fs::remove_file(&tmp_state);

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Domain filter
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_domain_filter() {
    let mut state = DaemonState::new();

    // Set domain filter BEFORE launch so Fetch.enable is called during
    // launch and the background fetch handler intercepts from the start.
    {
        let mut df = state.domain_filter.write().await;
        *df = Some(super::network::DomainFilter::new("example.com"));
    }

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // The active about:blank document must be patched immediately, not only
    // after a later navigation.
    let resp = execute_command(
        &json!({
            "id": "1-rtc", "action": "evaluate",
            "script": "(() => { try { new RTCPeerConnection({iceServers:[{urls:'stun:secret.blocked.com:3478'}]}); return 'NOT_BLOCKED'; } catch (error) { return error.name; } })()",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "SecurityError");

    // New tabs created after launch must receive the same controls before a
    // requested URL starts loading.
    let resp = execute_command(
        &json!({ "id": "1-tab", "action": "tab_new", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({
            "id": "1-tab-rtc", "action": "evaluate",
            "script": "(() => { try { new RTCPeerConnection({iceServers:[{urls:'stun:secret.blocked.com:3478'}]}); return 'NOT_BLOCKED'; } catch (error) { return error.name; } })()",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "SecurityError");

    let resp = execute_command(
        &json!({ "id": "1-tab-blocked", "action": "tab_new", "url": "https://blocked.com" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false);
    let error = resp["error"].as_str().unwrap_or("");
    assert!(
        error.contains("blocked.com") || error.contains("not allowed"),
        "Blocked tab URL should fail before loading, got: {}",
        error
    );

    let resp = execute_command(
        &json!({ "id": "1-window", "action": "window_new" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({
            "id": "1-window-rtc", "action": "evaluate",
            "script": "(() => { try { new RTCPeerConnection({iceServers:[{urls:'stun:secret.blocked.com:3478'}]}); return 'NOT_BLOCKED'; } catch (error) { return error.name; } })()",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "SecurityError");

    // Allowed domain
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Blocked domain
    let resp = execute_command(
        &json!({ "id": "3", "action": "navigate", "url": "https://blocked.com" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false);
    let error = resp["error"].as_str().unwrap();
    assert!(
        error.contains("blocked") || error.contains("not allowed"),
        "Should reject blocked domain, got: {}",
        error
    );

    // Verify that in-page fetch to a blocked domain is also blocked by
    // the Fetch interception layer (not just the navigate-level check).
    // First navigate to the allowed domain.
    let resp = execute_command(
        &json!({ "id": "4", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Attempt a cross-origin fetch to a blocked domain from the page.
    let resp = execute_command(
        &json!({
            "id": "5", "action": "evaluate",
            "script": "fetch('https://blocked.com/data').then(() => 'ok').catch(e => 'blocked:' + e.message)",
            "await": true,
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let result = get_data(&resp)["result"].as_str().unwrap_or("");
    assert!(
        result.starts_with("blocked:"),
        "Fetch to blocked domain should fail, got: {}",
        result,
    );

    // WebRTC uses DNS and UDP outside CDP Fetch interception, so the domain
    // filter must disable both Chromium constructor names before page scripts
    // can create a peer connection.
    let resp = execute_command(
        &json!({
            "id": "6", "action": "evaluate",
            "script": "['RTCPeerConnection','webkitRTCPeerConnection'].filter(name => typeof window[name] === 'function').map(name => { try { new window[name]({iceServers:[{urls:'stun:secret.blocked.com:3478'}]}); return name + ':NOT_BLOCKED'; } catch (error) { return name + ':' + error.name; } })",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let results = get_data(&resp)["result"].as_array().unwrap();
    assert!(
        !results.is_empty(),
        "RTCPeerConnection should be available in Chrome"
    );
    assert!(
        results.iter().all(|result| result
            .as_str()
            .is_some_and(|value| value.ends_with(":SecurityError"))),
        "Every peer connection constructor should be blocked, got: {:?}",
        results,
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_domain_filter_blocks_page_created_popup_before_first_request() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(Vec::<String>::new()));
    let requests_for_server = requests.clone();
    let server = tokio::spawn(async move {
        for _ in 0..20 {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let requests = requests_for_server.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let request_line = request.lines().next().unwrap_or("").to_string();
                if let Ok(mut logged) = requests.lock() {
                    logged.push(request_line);
                }

                let body = "<!doctype html><title>allowed</title><button id=\"go\">go</button>";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body,
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    let mut state = DaemonState::new();
    {
        let mut df = state.domain_filter.write().await;
        *df = Some(super::network::DomainFilter::new("localhost"));
    }

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": format!("http://localhost:{}/", port),
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "evaluate",
            "script": format!("window.open('http://127.0.0.1:{}/leak'); 'done'", port),
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let logged = requests.lock().unwrap().clone();
    assert!(
        !logged.iter().any(|line| line.contains(" /leak ")),
        "Blocked popup made a server request: {:?}",
        logged
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_domain_filter_blocks_service_worker_requests() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(Vec::<String>::new()));
    let requests_for_server = requests.clone();
    let server = tokio::spawn(async move {
        for _ in 0..20 {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let requests = requests_for_server.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let request_line = request.lines().next().unwrap_or("").to_string();
                if let Ok(mut logged) = requests.lock() {
                    logged.push(request_line.clone());
                }
                let path = request_line.split_whitespace().nth(1).unwrap_or("/");

                let (content_type, body) = if path == "/sw.js" {
                    (
                        "application/javascript",
                        format!(
                            r#"self.addEventListener('message', event => {{
    event.source && event.source.postMessage('started');
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 500);
    fetch('http://127.0.0.1:{}/leak', {{ mode: 'no-cors', signal: controller.signal }})
        .then(() => event.source && event.source.postMessage('leaked'))
        .catch(error => event.source && event.source.postMessage('blocked:' + error.name));
}});"#,
                            port
                        ),
                    )
                } else {
                    (
                        "text/html",
                        r#"<!doctype html><title>allowed</title><script>
window.swResult = 'pending';
navigator.serviceWorker.register('/sw.js').then(async reg => {
    await navigator.serviceWorker.ready;
    const sw = reg.active || reg.waiting || reg.installing;
    navigator.serviceWorker.addEventListener('message', event => {
        window.swResult = event.data;
    });
    sw.postMessage('go');
}).catch(error => {
    window.swResult = 'register:' + error.name;
});
</script>"#
                            .to_string(),
                    )
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    content_type,
                    body.len(),
                    body,
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    let mut state = DaemonState::new();
    {
        let mut df = state.domain_filter.write().await;
        *df = Some(super::network::DomainFilter::new("localhost"));
    }

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": format!("http://localhost:{}/", port),
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let mut sw_result = "pending".to_string();
    for _ in 0..50 {
        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "evaluate",
                "script": "window.swResult || 'pending'",
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        sw_result = get_data(&resp)["result"]
            .as_str()
            .unwrap_or("pending")
            .to_string();
        if !matches!(sw_result.as_str(), "pending" | "started") {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let logged = requests.lock().unwrap().clone();
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    server.abort();

    assert!(
        sw_result.starts_with("blocked:"),
        "Service worker request should be blocked, got: {}",
        sw_result
    );
    assert!(
        !logged.iter().any(|line| line.contains(" /leak ")),
        "Blocked service worker request reached the server: {:?}",
        logged
    );
}

#[tokio::test]
#[ignore]
async fn e2e_domain_filter_blocks_worker_websocket_requests() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(Vec::<String>::new()));
    let requests_for_server = requests.clone();
    let server = tokio::spawn(async move {
        for _ in 0..20 {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let requests = requests_for_server.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let request_line = request.lines().next().unwrap_or("").to_string();
                if let Ok(mut logged) = requests.lock() {
                    logged.push(request_line.clone());
                }
                let path = request_line.split_whitespace().nth(1).unwrap_or("/");

                let (content_type, body) = if path == "/worker.js" {
                    (
                        "application/javascript",
                        format!(
                            r#"self.addEventListener('message', async () => {{
    try {{
        const response = await fetch('/worker-ping');
        const text = await response.text();
        if (text !== 'pong') {{
            self.postMessage('fetch:' + text);
            return;
        }}
    }} catch (error) {{
        self.postMessage('fetch-error:' + error.name);
        return;
    }}

    try {{
        const ws = new WebSocket('ws://127.0.0.1:{}/leak');
        ws.onopen = () => self.postMessage('leaked');
        ws.onerror = () => self.postMessage('blocked:error');
    }} catch (error) {{
        self.postMessage('blocked:' + error.name);
    }}
}});"#,
                            port
                        ),
                    )
                } else if path == "/worker-ping" {
                    ("text/plain", "pong".to_string())
                } else {
                    (
                        "text/html",
                        r#"<!doctype html><title>allowed</title><script>
window.workerWsResult = 'pending';
const worker = new Worker('/worker.js');
worker.onmessage = event => {
    window.workerWsResult = event.data;
};
worker.onerror = () => {
    window.workerWsResult = 'worker:error';
};
worker.postMessage('go');
</script>"#
                            .to_string(),
                    )
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    content_type,
                    body.len(),
                    body,
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    let mut state = DaemonState::new();
    {
        let mut df = state.domain_filter.write().await;
        *df = Some(super::network::DomainFilter::new("localhost"));
    }

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": format!("http://localhost:{}/", port),
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let mut worker_result = "pending".to_string();
    for _ in 0..50 {
        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "evaluate",
                "script": "window.workerWsResult || 'pending'",
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        worker_result = get_data(&resp)["result"]
            .as_str()
            .unwrap_or("pending")
            .to_string();
        if worker_result != "pending" {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let logged = requests.lock().unwrap().clone();
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    server.abort();

    assert!(
        worker_result.starts_with("blocked:"),
        "Worker WebSocket should be blocked, got: {}",
        worker_result
    );
    assert!(
        !logged.iter().any(|line| line.contains(" /leak ")),
        "Blocked worker WebSocket reached the server: {:?}",
        logged
    );
}

#[tokio::test]
#[ignore]
async fn e2e_domain_filter_blocks_csp_self_worker_fallback() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(Vec::<String>::new()));
    let requests_for_server = requests.clone();
    let server = tokio::spawn(async move {
        for _ in 0..20 {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let requests = requests_for_server.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let request_line = request.lines().next().unwrap_or("").to_string();
                if let Ok(mut logged) = requests.lock() {
                    logged.push(request_line.clone());
                }
                let path = request_line.split_whitespace().nth(1).unwrap_or("/");

                let (content_type, body, csp) = if path == "/worker.js" {
                    (
                        "application/javascript",
                        format!(
                            r#"self.addEventListener('message', async () => {{
    try {{
        const response = await fetch('/worker-ping');
        const text = await response.text();
        try {{
            await fetch('http://127.0.0.1:{}/leak', {{ mode: 'no-cors' }});
            self.postMessage('leaked');
        }} catch (error) {{
            self.postMessage('started:' + text + ';blocked:' + error.name);
        }}
    }} catch (error) {{
        self.postMessage('blocked:' + error.name);
    }}
}});"#,
                            port
                        ),
                        None,
                    )
                } else if path == "/worker-ping" {
                    ("text/plain", "pong".to_string(), None)
                } else {
                    (
                        "text/html",
                        r#"<!doctype html><title>allowed</title><script>
window.workerCspResult = 'pending';
const worker = new Worker('/worker.js');
worker.onmessage = event => {
    window.workerCspResult = event.data;
};
worker.onerror = () => {
    window.workerCspResult = 'worker:error';
};
worker.postMessage('go');
</script>"#
                            .to_string(),
                        Some("Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; worker-src 'self'\r\n"),
                    )
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    content_type,
                    csp.unwrap_or(""),
                    body.len(),
                    body,
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    let mut state = DaemonState::new();
    {
        let mut df = state.domain_filter.write().await;
        *df = Some(super::network::DomainFilter::new("localhost"));
    }

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": format!("http://localhost:{}/", port),
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let mut worker_result = "pending".to_string();
    for _ in 0..50 {
        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "evaluate",
                "script": "window.workerCspResult || 'pending'",
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        worker_result = get_data(&resp)["result"]
            .as_str()
            .unwrap_or("pending")
            .to_string();
        if worker_result != "pending" {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let logged = requests.lock().unwrap().clone();
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    server.abort();

    assert_eq!(
        worker_result, "worker:error",
        "CSP-blocked worker bootstrap must fail closed instead of running an unguarded worker"
    );
    assert!(
        !logged.iter().any(|line| line.contains(" /leak ")),
        "Blocked CSP fallback worker request reached the server: {:?}",
        logged
    );
}

#[tokio::test]
#[ignore]
async fn e2e_domain_filter_blocks_module_worker_top_level_requests() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(Vec::<String>::new()));
    let requests_for_server = requests.clone();
    let server = tokio::spawn(async move {
        for _ in 0..20 {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let requests = requests_for_server.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let request_line = request.lines().next().unwrap_or("").to_string();
                if let Ok(mut logged) = requests.lock() {
                    logged.push(request_line.clone());
                }
                let path = request_line.split_whitespace().nth(1).unwrap_or("/");

                let (content_type, body) = if path == "/module-worker.js" {
                    (
                        "application/javascript",
                        format!(
                            r#"try {{
    await fetch('http://127.0.0.1:{}/leak', {{ mode: 'no-cors' }});
    self.postMessage('leaked');
}} catch (error) {{
    self.postMessage('blocked:' + error.name);
}}"#,
                            port
                        ),
                    )
                } else {
                    (
                        "text/html",
                        r#"<!doctype html><title>allowed</title><script>
window.moduleWorkerResult = 'pending';
const worker = new Worker('/module-worker.js', { type: 'module' });
worker.onmessage = event => {
    window.moduleWorkerResult = event.data;
};
worker.onerror = () => {
    window.moduleWorkerResult = 'worker:error';
};
</script>"#
                            .to_string(),
                    )
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    content_type,
                    body.len(),
                    body,
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    let mut state = DaemonState::new();
    {
        let mut df = state.domain_filter.write().await;
        *df = Some(super::network::DomainFilter::new("localhost"));
    }

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": format!("http://localhost:{}/", port),
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let mut worker_result = "pending".to_string();
    for _ in 0..50 {
        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "evaluate",
                "script": "window.moduleWorkerResult || 'pending'",
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        worker_result = get_data(&resp)["result"]
            .as_str()
            .unwrap_or("pending")
            .to_string();
        if worker_result != "pending" {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let logged = requests.lock().unwrap().clone();
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    server.abort();

    assert!(
        worker_result.starts_with("blocked:"),
        "Module worker top-level request should be blocked, got: {}",
        worker_result
    );
    assert!(
        !logged.iter().any(|line| line.contains(" /leak ")),
        "Blocked module worker request reached the server: {:?}",
        logged
    );
}

// ---------------------------------------------------------------------------
// Diff engine
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_diff_snapshot() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": native_test_fixture_url("snapshot_diff_probe")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Take a snapshot and use it as baseline for diff
    let resp = execute_command(&json!({ "id": "3", "action": "snapshot" }), &mut state).await;
    assert_success(&resp);
    let baseline = get_data(&resp)["snapshot"].as_str().unwrap().to_string();
    assert!(baseline.starts_with("- button \"Primary action\" [ref=e1]"));
    let baseline_dir = tempfile::tempdir().unwrap();
    let baseline_path = baseline_dir.path().join("baseline.txt");
    std::fs::write(&baseline_path, format!("{}\n", baseline)).unwrap();

    // A failed diff must preserve the refs from the last successful snapshot.
    let resp = execute_command(
        &json!({
            "id": "4",
            "action": "diff_snapshot",
            "baseline": baseline_path,
            "selector": "#missing"
        }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false);
    assert!(resp["error"]
        .as_str()
        .unwrap()
        .contains("did not match any element"));

    let resp = execute_command(
        &json!({ "id": "5", "action": "click", "selector": "e1" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Repeated diffs preserve document refs and report no content changes.
    for id in ["6", "7"] {
        let resp = execute_command(
            &json!({ "id": id, "action": "diff_snapshot", "baseline": baseline_path }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        let data = get_data(&resp);
        assert_eq!(data["changed"], false);
        assert_eq!(data["additions"], 0);
        assert_eq!(data["removals"], 0);
        assert_eq!(data["diff"], "");
    }

    // Modify the page
    let resp = execute_command(
        &json!({
            "id": "8",
            "action": "evaluate",
            "script": "document.querySelector('#primary-action').textContent = 'Updated action'"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Diff against baseline
    let resp = execute_command(
        &json!({ "id": "9", "action": "diff_snapshot", "baseline": baseline_path }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let data = get_data(&resp);
    assert_eq!(
        data["changed"], true,
        "Diff should detect the button change"
    );
    assert_eq!(data["additions"], 1);
    assert_eq!(data["removals"], 1);
    assert!(data["diff"].as_str().unwrap().contains("Updated action"));

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_diff_url_aligns_refs_after_snapshot() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let stale_url = "data:text/html,<button id='stale-action'>Stale action</button>";
    let url = native_test_fixture_url("snapshot_diff_probe");
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": stale_url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Populate refs that must be invalidated once the URL diff starts navigating.
    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "snapshot",
            "selector": "#stale-action"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(state.ref_map.get("e1").is_some());

    let resp = execute_command(
        &json!({
            "id": "4",
            "action": "diff_url",
            "url1": url,
            "url2": "http://[invalid"
        }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false);
    assert!(state.ref_map.entries_sorted().is_empty());

    let resp = execute_command(
        &json!({
            "id": "5",
            "action": "evaluate",
            "script": "document.querySelector('#primary-action')?.textContent"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "Primary action");

    let resp = execute_command(
        &json!({ "id": "6", "action": "click", "selector": "e1" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false);
    assert!(resp["error"].as_str().unwrap().contains("Unknown ref: e1"));

    // Populate the session ref map before comparing the same URL to itself.
    let resp = execute_command(&json!({ "id": "7", "action": "snapshot" }), &mut state).await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "8",
            "action": "diff_url",
            "url1": url,
            "url2": url
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let data = get_data(&resp);
    assert_eq!(data["diff"]["identical"], true);
    assert_eq!(data["diff"]["changed"], false);
    assert_eq!(data["diff"]["additions"], 0);
    assert_eq!(data["diff"]["removals"], 0);
    assert_ne!(
        data["snapshot1"], data["snapshot2"],
        "Replaced documents must not recycle actionable IDs"
    );
    assert_eq!(
        super::diff::snapshot_comparison_text(data["snapshot1"].as_str().unwrap()),
        super::diff::snapshot_comparison_text(data["snapshot2"].as_str().unwrap())
    );
    let primary_ref = state
        .ref_map
        .entries_sorted()
        .into_iter()
        .find(|(_, entry)| entry.name == "Primary action")
        .unwrap()
        .0;
    let next = execute_command(&json!({"id": "9", "action": "snapshot"}), &mut state).await;
    assert_success(&next);
    assert_eq!(
        get_data(&next)["refs"][&primary_ref]["name"],
        "Primary action"
    );
    assert_success(
        &execute_command(
            &json!({"id": "10", "action": "click", "selector": primary_ref}),
            &mut state,
        )
        .await,
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Phase 8 commands: focus, clear, count, boundingbox, innertext, setvalue
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_phase8_commands() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = concat!(
        "data:text/html,<html><body>",
        "<input id='a' value='original'>",
        "<input id='b' value='other'>",
        "<p class='item'>One</p>",
        "<p class='item'>Two</p>",
        "<p class='item'>Three</p>",
        "<div id='box' style='width:200px;height:100px;background:red'>Box</div>",
        "</body></html>"
    );

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Focus
    let resp = execute_command(
        &json!({ "id": "10", "action": "focus", "selector": "#a" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Clear
    let resp = execute_command(
        &json!({ "id": "11", "action": "clear", "selector": "#a" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "12", "action": "evaluate", "script": "document.getElementById('a').value" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "");

    // Set value
    let resp = execute_command(
        &json!({ "id": "13", "action": "setvalue", "selector": "#b", "value": "new-value" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "14", "action": "inputvalue", "selector": "#b" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["value"], "new-value");

    // Count
    let resp = execute_command(
        &json!({ "id": "15", "action": "count", "selector": ".item" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["count"], 3);

    // Bounding box
    let resp = execute_command(
        &json!({ "id": "16", "action": "boundingbox", "selector": "#box" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let bbox = get_data(&resp);
    assert_eq!(bbox["width"], 200.0);
    assert_eq!(bbox["height"], 100.0);
    assert!(bbox["x"].as_f64().is_some());
    assert!(bbox["y"].as_f64().is_some());

    // Inner text
    let resp = execute_command(
        &json!({ "id": "17", "action": "innertext", "selector": "#box" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["text"], "Box");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Auto-launch (tests that commands auto-launch when no browser exists)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_auto_launch() {
    let mut state = DaemonState::new();

    // Navigate without explicit launch -- should auto-launch
    let resp = execute_command(
        &json!({ "id": "1", "action": "navigate", "url": "data:text/html,<h1>Auto</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(state.browser.is_some(), "Browser should be auto-launched");

    let resp = execute_command(
        &json!({ "id": "2", "action": "evaluate", "script": "document.querySelector('h1').textContent" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "Auto");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_error_handling() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<h1>Errors</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Unknown action
    let resp = execute_command(
        &json!({ "id": "10", "action": "nonexistent_action" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false);
    assert!(resp["error"]
        .as_str()
        .unwrap()
        .contains("Not yet implemented"));

    // Missing required parameter
    let resp = execute_command(
        &json!({ "id": "11", "action": "fill", "selector": "#x" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false);
    assert!(resp["error"].as_str().unwrap().contains("value"));

    // Click on non-existent element
    let resp = execute_command(
        &json!({ "id": "12", "action": "click", "selector": "#does-not-exist" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false);

    // Evaluate syntax error
    let resp = execute_command(
        &json!({ "id": "13", "action": "evaluate", "script": "}{invalid" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false);
    assert!(resp["error"].as_str().unwrap().contains("error"));

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_click_reports_covering_overlay() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = r#"
        <html>
        <body>
            <button id="target" onclick="document.getElementById('result').textContent = 'clicked'">
                Target
            </button>
            <div id="consent-banner" style="position:fixed;inset:0;z-index:10;background:rgba(0,0,0,0.1)">
                <button id="dismiss" style="position:absolute;right:20px;bottom:20px"
                    onclick="document.getElementById('consent-banner').remove()">
                    Dismiss
                </button>
            </div>
            <div id="result">idle</div>
        </body>
        </html>
    "#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "click", "selector": "#target" }),
        &mut state,
    )
    .await;
    assert_eq!(resp["success"], false, "covered target should fail: {resp}");
    let error = resp["error"].as_str().unwrap_or_default();
    assert!(
        error.contains("covered by <div#consent-banner>"),
        "unexpected covered-click error: {}",
        error
    );

    let resp = execute_command(
        &json!({ "id": "4", "action": "gettext", "selector": "#result" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["text"], "idle");

    let resp = execute_command(
        &json!({ "id": "5", "action": "click", "selector": "#dismiss" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "6", "action": "click", "selector": "#target" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "7", "action": "gettext", "selector": "#result" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["text"], "clicked");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Profile cookie persistence across restarts
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_profile_cookie_persistence() {
    let profile_dir = std::env::temp_dir().join(format!(
        "agent-browser-e2e-profile-{}",
        uuid::Uuid::new_v4()
    ));

    // Session 1: launch with profile, set a cookie, close
    {
        let mut state = DaemonState::new();

        let resp = execute_command(
            &json!({
                "id": "1",
                "action": "launch",
                "headless": true,
                "profile": profile_dir.to_str().unwrap()
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "cookies_set",
                "name": "persist_test",
                "value": "should_survive_restart",
                "domain": ".example.com",
                "path": "/",
                "expires": 2000000000
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        // Verify cookie is set
        let resp =
            execute_command(&json!({ "id": "4", "action": "cookies_get" }), &mut state).await;
        assert_success(&resp);
        let cookies = get_data(&resp)["cookies"].as_array().unwrap();
        let found = cookies
            .iter()
            .any(|c| c["name"] == "persist_test" && c["value"] == "should_survive_restart");
        assert!(found, "Cookie should exist before close");

        let resp = execute_command(&json!({ "id": "5", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Session 2: reopen with the same profile, verify cookie persisted
    {
        let mut state = DaemonState::new();

        let resp = execute_command(
            &json!({
                "id": "10",
                "action": "launch",
                "headless": true,
                "profile": profile_dir.to_str().unwrap()
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "11", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp =
            execute_command(&json!({ "id": "12", "action": "cookies_get" }), &mut state).await;
        assert_success(&resp);
        let cookies = get_data(&resp)["cookies"].as_array().unwrap();
        let found = cookies
            .iter()
            .any(|c| c["name"] == "persist_test" && c["value"] == "should_survive_restart");
        assert!(
            found,
            "Cookie should persist across restart with --profile. Cookies found: {:?}",
            cookies
                .iter()
                .map(|c| c["name"].as_str().unwrap_or("?"))
                .collect::<Vec<_>>()
        );

        let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    let _ = std::fs::remove_dir_all(&profile_dir);
}

// ---------------------------------------------------------------------------
// Inspect / CDP URL
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_get_cdp_url() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "2", "action": "cdp_url" }), &mut state).await;
    assert_success(&resp);
    let cdp_url = get_data(&resp)["cdpUrl"]
        .as_str()
        .expect("cdpUrl should be a string");
    assert!(
        cdp_url.starts_with("ws://"),
        "CDP URL should start with ws://, got: {}",
        cdp_url
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_inspect() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "3", "action": "inspect" }), &mut state).await;
    assert_success(&resp);
    let data = get_data(&resp);
    assert_eq!(data["opened"], true);
    let url = data["url"]
        .as_str()
        .expect("inspect url should be a string");
    assert!(
        url.starts_with("http://127.0.0.1:"),
        "Inspect URL should be http://127.0.0.1:<port>, got: {}",
        url
    );

    // Verify the HTTP redirect serves a 302 to the DevTools frontend
    let http_resp = reqwest::get(url).await;
    match http_resp {
        Ok(r) => {
            let final_url = r.url().to_string();
            assert!(
                final_url.contains("devtools/devtools_app.html"),
                "Redirect should point to DevTools frontend, got: {}",
                final_url
            );
        }
        Err(e) => {
            panic!("HTTP GET to inspect URL failed: {}", e);
        }
    }

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Stale ref fallback (#805): clicking a ref after the DOM has been replaced
// should fall back to role/name lookup instead of failing.
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_click_stale_ref_falls_back_to_role_name() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate to a page with a button that replaces the DOM when clicked.
    let html = r#"data:text/html,<body>
        <div id="c">
            <button onclick="
                var c = document.getElementById('c');
                c.innerHTML = '';
                var b = document.createElement('button');
                b.textContent = 'Target';
                b.onclick = function() { document.title = 'clicked'; };
                c.appendChild(b);
                document.title = 'replaced';
            ">Replace</button>
            <button>Target</button>
        </div>
    </body>"#;

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Snapshot to populate the ref_map with backend_node_ids.
    let resp = execute_command(&json!({ "id": "3", "action": "snapshot" }), &mut state).await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap();
    assert!(
        snapshot.contains("Replace"),
        "Snapshot should contain Replace button"
    );
    assert!(
        snapshot.contains("Target"),
        "Snapshot should contain Target button"
    );

    // Click "Replace" — this removes all DOM nodes and recreates them,
    // making the backend_node_id for "Target" stale.
    let resp = execute_command(
        &json!({ "id": "4", "action": "click", "selector": "e1" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Verify the DOM was actually replaced.
    let resp = execute_command(&json!({ "id": "5", "action": "title" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["title"], "replaced");

    // Now click the stale "Target" ref. Before the fix this returned:
    //   "CDP error (DOM.getBoxModel): Could not compute box model."
    // After the fix it falls back to role/name lookup and succeeds.
    let resp = execute_command(
        &json!({ "id": "6", "action": "click", "selector": "e2" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Verify the fallback click hit the right (recreated) button.
    let resp = execute_command(&json!({ "id": "7", "action": "title" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["title"],
        "clicked",
        "Stale ref should have been resolved via role/name fallback"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Regression: Material Design checkbox/radio (#832)
//
// Material Design controls hide the native <input> off-screen and place
// overlay elements (ripple, touch-target) on top.  Coordinate-based CDP
// clicks may therefore miss the actual input.  The check/uncheck actions
// must detect this and fall back to a JS .click() — matching the behaviour
// that Playwright provided in v0.19.0.
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_material_checkbox_check_uncheck() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Inline HTML that reproduces the Material Design DOM pattern:
    // - Native <input> is visually hidden (position:absolute, opacity:0, off-screen)
    // - A ripple overlay sits on top with pointer-events:all, intercepting coordinate clicks
    // - An ARIA-only checkbox uses role="checkbox" + aria-checked (no native input)
    let html = concat!(
        "data:text/html,<html><body>",
        // -- Native baseline --
        "<input id='native' type='checkbox'>",
        // -- Material-style hidden-input checkbox --
        "<div id='mat' style='position:relative;padding:12px'>",
          "<input id='mat-input' type='checkbox' style='position:absolute;opacity:0;width:1px;height:1px;top:-9999px;left:-9999px;pointer-events:none'>",
          "<div style='position:absolute;top:0;left:0;width:48px;height:48px;pointer-events:all;z-index:10'></div>",
          "<span>Material CB</span>",
        "</div>",
        // -- ARIA-only checkbox (no native input) --
        "<div id='aria' role='checkbox' aria-checked='false' tabindex='0'>ARIA CB</div>",
        "<script>",
          "document.getElementById('aria').addEventListener('click',function(){",
            "var c=this.getAttribute('aria-checked')==='true';",
            "this.setAttribute('aria-checked',String(!c));",
          "});",
        "</script>",
        "</body></html>"
    );

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // ---- Native checkbox (sanity baseline) ----
    let resp = execute_command(
        &json!({ "id": "10", "action": "ischecked", "selector": "#native" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["checked"], false);

    let resp = execute_command(
        &json!({ "id": "11", "action": "check", "selector": "#native" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "12", "action": "ischecked", "selector": "#native" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["checked"], true, "native check failed");

    // ---- Material checkbox (hidden input + overlay) ----
    // ischecked on the wrapper should detect the nested hidden input's state
    let resp = execute_command(
        &json!({ "id": "20", "action": "ischecked", "selector": "#mat" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["checked"], false);

    let resp = execute_command(
        &json!({ "id": "21", "action": "check", "selector": "#mat" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "22", "action": "ischecked", "selector": "#mat" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["checked"],
        true,
        "Material checkbox should be checked after check action (#832)"
    );

    // Idempotency: check again should be a no-op
    let resp = execute_command(
        &json!({ "id": "23", "action": "check", "selector": "#mat" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "24", "action": "ischecked", "selector": "#mat" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["checked"],
        true,
        "Material checkbox should stay checked on redundant check"
    );

    // Uncheck
    let resp = execute_command(
        &json!({ "id": "25", "action": "uncheck", "selector": "#mat" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "26", "action": "ischecked", "selector": "#mat" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["checked"],
        false,
        "Material checkbox should be unchecked after uncheck action"
    );

    // ---- ARIA-only checkbox ----
    let resp = execute_command(
        &json!({ "id": "30", "action": "ischecked", "selector": "#aria" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["checked"], false);

    let resp = execute_command(
        &json!({ "id": "31", "action": "check", "selector": "#aria" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "32", "action": "ischecked", "selector": "#aria" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["checked"],
        true,
        "ARIA checkbox should be checked after check action"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Issue #841 – snapshot -C and screenshot --annotate must not hang over WSS
// (PS: -C is deprecated, cursor-interactive elements are referred by default now)
// ---------------------------------------------------------------------------

/// Verifies that `snapshot` detects elements with cursor:pointer / onclick / tabindex,
/// produces the correct v0.19.0-compatible output format, deduplicates against the ARIA
/// tree, and completes in bounded time (no sequential CDP round-trip explosion).
#[tokio::test]
#[ignore]
async fn e2e_snapshot_cursor_interactive() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Page with:
    //  - <button> and <a> (standard interactive – ARIA tree)
    //  - <div cursor:pointer onclick> (clickable – cursor section)
    //  - <div tabindex=0> (focusable – cursor section)
    //  - <span cursor:pointer> (clickable – cursor section)
    //  - <span cursor:pointer> child of <div cursor:pointer> (inherited – skip)
    let html = concat!(
        "<html><body>",
        "<a href='#'>Link</a>",
        "<button>Btn</button>",
        "<div style='cursor:pointer' onclick='x()'>ClickDiv</div>",
        "<div tabindex='0'>FocusDiv</div>",
        "<span style='cursor:pointer'>PointerSpan</span>",
        "<div style='cursor:pointer'><span>InheritChild</span></div>",
        "</body></html>",
    );

    let resp = execute_command(
        &json!({ "id": "2", "action": "setcontent", "html": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // snapshot -i: interactive tree
    let start = std::time::Instant::now();
    let resp = execute_command(
        &json!({ "id": "3", "action": "snapshot", "interactive": true }),
        &mut state,
    )
    .await;
    let elapsed = start.elapsed();
    assert_success(&resp);

    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap();

    // v0.19.0 output format: role + hints
    assert!(
        snapshot.contains("clickable") && snapshot.contains("[cursor:pointer"),
        "Expected v0.19.0-format cursor output with hints:\n{}",
        snapshot,
    );

    // Role differentiation: tabindex-only → focusable
    assert!(
        snapshot.contains("focusable") && snapshot.contains("[tabindex]"),
        "Expected focusable role for tabindex-only element:\n{}",
        snapshot,
    );

    // Text dedup: "Link" and "Btn" are in the ARIA tree, so must NOT suffix
    // with cursor-interactive info. Verify line by line.
    for line in snapshot.lines() {
        assert!(
            !(line.contains("\"Link\"")
                && (line.contains("clickable")
                    || line.contains("focusable")
                    || line.contains("editable"))),
            "Standard <a> element should not have cursor-interactive info:\n{}",
            line
        );
        assert!(
            !(line.contains("\"Btn\"")
                && (line.contains("clickable")
                    || line.contains("focusable")
                    || line.contains("editable"))),
            "Standard <button> element should not have cursor-interactive info:\n{}",
            line
        );
    }

    // Must complete quickly (< 5s), not hit the 30s CDP timeout
    assert!(
        elapsed.as_secs() < 5,
        "snapshot took {:?}, expected < 5s (Issue #841 regression)",
        elapsed,
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Verifies that `screenshot --annotate` completes in bounded time even with
/// many interactive elements. Guards against the sequential CDP round-trip
/// regression that caused hangs over high-latency WSS (Issue #841).
#[tokio::test]
#[ignore]
async fn e2e_screenshot_annotate_many_elements() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // 50 buttons: old sequential code would do 50×2×200ms ≈ 20s over WSS.
    let mut html = String::from("<html><body>");
    for i in 1..=50 {
        html.push_str(&format!("<button>Button {}</button>", i));
    }
    html.push_str("</body></html>");

    let resp = execute_command(
        &json!({ "id": "2", "action": "setcontent", "html": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let start = std::time::Instant::now();
    let resp = execute_command(
        &json!({ "id": "3", "action": "screenshot", "annotate": true }),
        &mut state,
    )
    .await;
    let elapsed = start.elapsed();
    assert_success(&resp);

    let annotations = get_data(&resp)["annotations"]
        .as_array()
        .expect("Annotated screenshot should return annotations");

    assert!(
        annotations.len() >= 50,
        "Expected at least 50 annotations, got {}",
        annotations.len(),
    );

    // Must complete quickly (< 10s), not hit the 30s CDP timeout
    assert!(
        elapsed.as_secs() < 10,
        "screenshot --annotate with 50 elements took {:?}, expected < 10s (Issue #841)",
        elapsed,
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Verifies `snapshot` with many cursor-interactive elements completes in
/// bounded time. Direct regression test for Issue #841's root cause: N×2
/// sequential CDP round-trips per cursor-interactive element.
#[tokio::test]
#[ignore]
async fn e2e_snapshot_cursor_many_elements() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // 100 cursor-interactive divs: old code = 200 sequential CDP calls,
    // at 200ms WSS latency = 40s timeout. New code must finish in seconds.
    let mut html = String::from("<html><body>");
    for i in 1..=100 {
        html.push_str(&format!(
            "<div style='cursor:pointer' onclick='x()'>Item {}</div>",
            i,
        ));
    }
    html.push_str("</body></html>");

    let resp = execute_command(
        &json!({ "id": "2", "action": "setcontent", "html": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let start = std::time::Instant::now();
    let resp = execute_command(
        &json!({ "id": "3", "action": "snapshot", "interactive": true }),
        &mut state,
    )
    .await;
    let elapsed = start.elapsed();
    assert_success(&resp);

    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap();

    // All 100 items should appear
    assert!(
        snapshot.contains("Item 1") && snapshot.contains("Item 100"),
        "Expected all 100 cursor-interactive items in output",
    );

    // All should have v0.19.0-format hints
    assert!(
        snapshot.contains("[cursor:pointer, onclick]"),
        "Expected v0.19.0-format hints",
    );

    // Must complete quickly
    assert!(
        elapsed.as_secs() < 10,
        "snapshot with 100 cursor elements took {:?}, expected < 10s (Issue #841)",
        elapsed,
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Test that a selector-scoped snapshot of a web component includes the
/// content of its shadow root.
#[tokio::test]
#[ignore]
async fn e2e_snapshot_selector_includes_shadow_root() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html =
        "data:text/html,<my-card><span>Slotted</span></my-card><button>Outside</button><script>\
        customElements.define('my-card', class extends HTMLElement { constructor() { \
        super(); this.attachShadow({ mode: 'open' }).innerHTML = \
        '<button>Inside shadow</button><slot></slot>'; } });</script>";

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "snapshot", "selector": "my-card" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap();

    assert!(
        snapshot.contains("button \"Inside shadow\""),
        "Snapshot should contain the shadow root button: {}",
        snapshot
    );
    assert!(
        snapshot.contains("Slotted"),
        "Snapshot should contain the slotted text: {}",
        snapshot
    );
    assert!(
        !snapshot.contains("Outside"),
        "Snapshot should not contain content outside the selector: {}",
        snapshot
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Test that InlineTextBox nodes are filtered from snapshot output while preserving
/// the actual text content from parent elements.
#[tokio::test]
#[ignore]
async fn e2e_snapshot_continuous_static_text() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Simple HTML with text content that would generate InlineTextBox nodes and sperate to multiple StaticText nodes
    let html =
        "data:text/html,<html><body><div><span>Hello</span> <span>World</span></div></body></html>";

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Take snapshot to capture full output and verify InlineTextBox filtering and StaticText aggregation
    let start = std::time::Instant::now();
    let resp = execute_command(&json!({ "id": "3", "action": "snapshot" }), &mut state).await;
    assert_success(&resp);
    let elapsed = start.elapsed();

    let snapshot_output = get_data(&resp)["snapshot"].as_str().unwrap();

    // Verify that InlineTextBox does not appear in the output
    assert!(
        !snapshot_output.contains("InlineTextBox"),
        "Snapshot output should not contain InlineTextBox: {}",
        snapshot_output
    );

    // Verify that the actual text content is preserved
    assert!(
        snapshot_output.contains("Hello World"),
        "Snapshot should contain 'Hello World': {}",
        snapshot_output
    );

    // Must complete quickly
    assert!(
        elapsed.as_secs() < 5,
        "snapshot with InlineTextBox filtering took {:?}, expected < 5s",
        elapsed,
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Test that a selector-scoped snapshot renders each element once when plain
/// wrapper elements (ignored in the AX tree) sit between the matched element
/// and its descendants.
#[tokio::test]
#[ignore]
async fn e2e_snapshot_selector_no_duplicates() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = "data:text/html,<main><div><div><button>Save</button></div></div>\
        <select><option>Small</option><option>Large</option></select></main>\
        <footer><button>Outside</button></footer>";

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "snapshot", "selector": "main" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap();

    for name in ["button \"Save\"", "option \"Small\"", "option \"Large\""] {
        assert_eq!(
            snapshot.matches(name).count(),
            1,
            "{} should appear once: {}",
            name,
            snapshot
        );
    }
    assert!(
        !snapshot.contains("Outside"),
        "Snapshot should not contain content outside the selector: {}",
        snapshot
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Helper: tiny HTTP server that echoes request headers as JSON
// ---------------------------------------------------------------------------

/// Starts a TCP listener on localhost:0 and spawns a task that accepts
/// connections, reads the HTTP request, and responds with a JSON body
/// containing all received request headers. Returns the server's base URL.
async fn start_echo_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base_url = format!("http://127.0.0.1:{}", port);

    let handle = tokio::spawn(async move {
        // Serve up to 20 requests then exit (enough for all tests).
        for _ in 0..20 {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);

                // Parse headers from the HTTP request.
                let mut headers = serde_json::Map::new();
                for line in request.lines().skip(1) {
                    if line.is_empty() {
                        break;
                    }
                    if let Some((key, value)) = line.split_once(": ") {
                        headers.insert(key.to_string(), Value::String(value.to_string()));
                    }
                }

                let body = serde_json::to_string(&json!({ "headers": headers })).unwrap();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Access-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\
                     Connection: close\r\n\r\n{}",
                    body.len(),
                    body,
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    (base_url, handle)
}

async fn start_webmcp_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut buffer = vec![0u8; 4096];
                let _ = stream.read(&mut buffer).await;
                let request = String::from_utf8_lossy(&buffer);
                let body = if request.starts_with("GET /frame.html ") {
                    native_test_fixture_html("webmcp_frame_probe")
                        .replace("__PORT__", &port.to_string())
                } else if request.starts_with("GET /context.html ") {
                    native_test_fixture_html("webmcp_context_probe").to_string()
                } else if request.starts_with("GET /empty.html ") {
                    "<!doctype html><title>No page tools</title>".to_string()
                } else if request.starts_with("GET /delayed.html ") {
                    native_test_fixture_html("webmcp_delayed_probe").to_string()
                } else {
                    native_test_fixture_html("webmcp_probe").replace("__PORT__", &port.to_string())
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    });
    (format!("http://127.0.0.1:{port}"), handle)
}

/// Starts a tiny cookie-gated app that behaves like a Next dev target with
/// cookie-backed login state.
async fn start_cookie_login_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base_url = format!("http://127.0.0.1:{}", port);

    let handle = tokio::spawn(async move {
        for _ in 0..100 {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };

            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let request_line = request.lines().next().unwrap_or_default();
                let path = request_line.split_whitespace().nth(1).unwrap_or("/");
                let has_auth_cookie = request.lines().any(|line| {
                    line.to_ascii_lowercase().starts_with("cookie:")
                        && line.contains("next_dev_loop_auth=1")
                });

                let mut headers = vec!["Content-Type: text/html".to_string()];
                let body = if path.starts_with("/login") {
                    headers.push(
                        "Set-Cookie: next_dev_loop_auth=1; Path=/; Max-Age=3600; SameSite=Lax"
                            .to_string(),
                    );
                    "<!doctype html><title>Logged in</title><main>Login complete</main>".to_string()
                } else if has_auth_cookie {
                    "<!doctype html><title>Home</title><main>Welcome back</main>".to_string()
                } else {
                    "<!doctype html><title>Home</title><main>Please sign in</main>".to_string()
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\n{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    headers.join("\r\n"),
                    body.len(),
                    body,
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    (base_url, handle)
}

/// Starts a tiny HTTP server that serves a delayed-render login form.
///
/// The page continuously fetches `/ping` so `networkidle` is hard to reach,
/// while the login form itself appears after `render_delay_ms`.
async fn start_delayed_login_server(
    render_delay_ms: u64,
    ping_interval_ms: u64,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base_url = format!("http://127.0.0.1:{}", port);

    let handle = tokio::spawn(async move {
        // Serve enough requests for navigation + many background /ping calls.
        for _ in 0..1000 {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };

            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let request_line = request.lines().next().unwrap_or_default();
                let path = request_line.split_whitespace().nth(1).unwrap_or("/");

                let (status, content_type, body) = if path.starts_with("/ping") {
                    ("204 No Content", "text/plain", String::new())
                } else {
                    let html = format!(
                        r#"<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Delayed Login</title></head>
  <body>
    <input id="search" type="text" name="search" />
    <div id="root">loading...</div>
    <script>
      setInterval(() => {{
        fetch('/ping?ts=' + Date.now()).catch(() => {{}});
      }}, {ping_interval_ms});

      setTimeout(() => {{
        const root = document.getElementById('root');
        root.innerHTML = `
          <form id="login-form">
            <input type="email" name="email" />
            <input type="password" name="password" />
            <button type="submit">Sign in</button>
          </form>
        `;
        document.getElementById('login-form').addEventListener('submit', function(e) {{
          e.preventDefault();
          e.stopPropagation();
          window.__submitted = true;
        }});
      }}, {render_delay_ms});
    </script>
  </body>
</html>"#,
                    );
                    ("200 OK", "text/html", html)
                };

                let response = format!(
                    "HTTP/1.1 {}\r\nContent-Type: {}\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    content_type,
                    body.len(),
                    body,
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    (base_url, handle)
}

/// Starts a stateful login page whose form appears only after an in-page
/// action. The counter covers top-level documents so tests can prove that
/// no-navigation login did not reload or replace the page.
async fn start_stateful_auth_login_server(
) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base_url = format!("http://127.0.0.1:{}", port);
    let document_requests = Arc::new(AtomicUsize::new(0));
    let counter = document_requests.clone();

    let handle = tokio::spawn(async move {
        for _ in 0..100 {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let counter = counter.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                if !path.starts_with("/favicon") {
                    counter.fetch_add(1, Ordering::SeqCst);
                }

                let body = r#"<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Stateful Login</title><link rel="icon" href="data:," /></head>
  <body>
    <button id="reveal" type="button">Continue to login</button>
    <form id="login-form" style="display:none">
      <input type="email" name="email" />
      <input type="password" name="password" />
      <button type="submit">Sign in</button>
    </form>
    <script>
      document.getElementById('reveal').addEventListener('click', () => {
        document.getElementById('login-form').style.display = 'block';
        window.__revealed = true;
      });
      document.getElementById('login-form').addEventListener('submit', (event) => {
        event.preventDefault();
        window.__submitted = true;
      });
    </script>
  </body>
</html>"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body,
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    (base_url, document_requests, handle)
}

fn unique_auth_profile_name(suffix: &str) -> String {
    format!(
        "e2e-auth-login-{}-{}",
        suffix,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_nanos()
    )
}

#[tokio::test]
#[ignore]
async fn e2e_auth_login_selects_usable_controls_and_preserves_credential_targets() {
    let (base_url, _, server) = start_stateful_auth_login_server().await;
    let mut state = DaemonState::new();
    let profile = unique_auth_profile_name("changed-target");
    for command in [
        json!({ "id": "launch", "action": "launch", "headless": true }),
        json!({ "id": "open", "action": "navigate", "url": base_url }),
        json!({ "id": "save", "action": "auth_save", "name": profile,
            "url": base_url, "username": "target@example.test", "password": "target-password" }),
    ] {
        assert_success(&execute_command(&command, &mut state).await);
    }
    let mut results = Vec::new();
    for field in ["user", "pass"] {
        for phase in [
            "normal",
            "selection",
            "focus-detach",
            "focus-redirect",
            "select-redirect",
            "focus-readonly",
            "input-detach",
            "exception",
            "textarea",
            "contenteditable",
            "no-native-edit",
            "no-native-edit-textarea",
            "no-native-edit-contenteditable",
        ] {
            let script = format!(
                r#"(() => {{
                    document.body.innerHTML = '<form><input id="decoy" value="decoy:"><input id="hidden-user" type="email" autocomplete="username" hidden><input id="user" type="email" autocomplete="username webauthn"><input id="hidden-pass" type="password" hidden><input id="pass" type="password"><button hidden type="submit" onclick="window.hiddenClicked = true">Hidden</button><button type="submit" onclick="window.visibleClicked = true">Sign in</button></form>';
                    if ('{phase}'.endsWith('textarea')) document.getElementById('{field}').outerHTML = '<textarea id="{field}">old value</textarea>';
                    if ('{phase}'.endsWith('contenteditable')) document.getElementById('{field}').outerHTML = '<div id="{field}" contenteditable="true">old value</div>';
                    window.events = [];
                    window.submitted = false;
                    window.hiddenClicked = false;
                    window.visibleClicked = false;
                    window.nativeEdit ??= document.execCommand;
                    document.execCommand = '{phase}'.startsWith('no-native-edit') ? undefined : window.nativeEdit;
                    const decoy = document.getElementById('decoy');
                    const target = document.getElementById('{field}');
                    const replace = () => {{
                        window.detached = target;
                        target.replaceWith(target.cloneNode(true));
                        decoy.focus();
                    }};
                    window.detached = null;
                    document.querySelector('form').onsubmit = event => {{
                        event.preventDefault();
                        window.submitted = true;
                    }};
                    document.oninput = event => window.events.push({{
                        id: event.target.id, value: event.target.value ?? event.target.textContent,
                        trusted: event.isTrusted
                    }});
                    decoy.focus();
                    decoy.setSelectionRange(6, 6);
                    if ('{phase}' === 'selection') {{
                        const box = target.getBoundingClientRect.bind(target);
                        target.getBoundingClientRect = () => {{
                            const rect = box();
                            queueMicrotask(replace);
                            return rect;
                        }};
                    }}
                    if ('{phase}' === 'focus-detach') target.onfocus = replace;
                    if ('{phase}' === 'focus-redirect') target.onfocus = () => decoy.focus();
                    if ('{phase}' === 'select-redirect') target.select = () => decoy.focus();
                    if ('{phase}' === 'focus-readonly') target.onfocus = () => target.readOnly = true;
                    if ('{phase}' === 'input-detach') target.oninput = replace;
                    if ('{phase}' === 'exception') target.focus = () => {{ throw new Error('page failure'); }};
                }})()"#
            );
            assert_success(
                &execute_command(
                    &json!({ "id": "fixture", "action": "evaluate", "script": script }),
                    &mut state,
                )
                .await,
            );
            let mut command = json!({
                "id": "login", "action": "auth_login", "name": profile,
                "noNavigate": true, "timeout": 200
            });
            if phase.ends_with("textarea") || phase.ends_with("contenteditable") {
                command["usernameSelector"] = json!("#user");
                command["passwordSelector"] = json!("#pass");
            }
            let login = execute_command(&command, &mut state).await;
            let observation = execute_command(
                &json!({
                    "id": "observe", "action": "evaluate",
                    "script": "({ user: document.getElementById('user').value ?? document.getElementById('user').textContent, pass: document.getElementById('pass').value ?? document.getElementById('pass').textContent, decoy: document.getElementById('decoy').value, events: window.events, hiddenUser: document.getElementById('hidden-user').value, hiddenPass: document.getElementById('hidden-pass').value, hiddenClicked: window.hiddenClicked, visibleClicked: window.visibleClicked, submitted: window.submitted, detached: !!window.detached && !window.detached.isConnected })"
                }),
                &mut state,
            )
            .await;
            results.push((field, phase, login, observation));
        }
    }
    for command in [
        json!({ "id": "delete", "action": "auth_delete", "name": profile }),
        json!({ "id": "close", "action": "close" }),
    ] {
        assert_success(&execute_command(&command, &mut state).await);
    }
    server.abort();
    for (field, phase, login, observation) in results {
        assert_success(&observation);
        let observed = &get_data(&observation)["result"];
        let normal = ["normal", "textarea", "contenteditable"].contains(&phase)
            || phase.starts_with("no-native-edit");
        assert_eq!(observed["decoy"], "decoy:", "{field}/{phase}: {observed}");
        assert_eq!(observed["hiddenUser"], "", "{field}/{phase}: {observed}");
        assert_eq!(observed["hiddenPass"], "", "{field}/{phase}: {observed}");
        assert_eq!(
            observed["hiddenClicked"], false,
            "{field}/{phase}: {observed}"
        );
        assert_eq!(
            observed["visibleClicked"], normal,
            "{field}/{phase}: {observed}"
        );
        assert_eq!(login["success"], normal, "{field}/{phase}: {login}");
        assert_eq!(observed["submitted"], normal, "{field}/{phase}: {observed}");
        if phase == "selection" || phase == "focus-detach" || phase == "input-detach" {
            assert_eq!(observed["detached"], true, "{field}/{phase}: {observed}");
        }
        for event in observed["events"].as_array().unwrap() {
            let expected = match event["id"].as_str().unwrap() {
                "user" => "target@example.test",
                "pass" => "target-password",
                other => panic!("{field}/{phase}: input delivered to {other}: {event}"),
            };
            assert_eq!(event["value"], expected, "{field}/{phase}: {event}");
            assert_eq!(
                event["trusted"],
                !phase.starts_with("no-native-edit"),
                "{field}/{phase}: {event}"
            );
        }
        let expected_user = if normal || field == "pass" || phase == "input-detach" {
            "target@example.test"
        } else {
            ""
        };
        let expected_pass = if normal || (field == "pass" && phase == "input-detach") {
            "target-password"
        } else {
            ""
        };
        assert_eq!(
            observed["user"], expected_user,
            "{field}/{phase}: {observed}"
        );
        assert_eq!(
            observed["pass"], expected_pass,
            "{field}/{phase}: {observed}"
        );
    }
}

#[tokio::test]
#[ignore]
async fn e2e_auth_login_no_navigate_preserves_active_page_state() {
    let (base_url, document_requests, _server) = start_stateful_auth_login_server().await;
    let mut state = DaemonState::new();
    let profile_name = unique_auth_profile_name("no-navigate");

    assert_success(
        &execute_command(
            &json!({ "id": "1", "action": "launch", "headless": true }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "2", "action": "navigate", "url": format!("{}/flow/start?state=ready#login", base_url) }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "3", "action": "click", "selector": "#reveal" }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "4", "action": "evaluate", "script": "window.__marker = 'preserved'" }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({
                "id": "5",
                "action": "auth_save",
                "name": profile_name.clone(),
                "url": format!("{}/credentials/login?credential-query-sentinel#credential-fragment-sentinel", base_url),
                "username": "stateful-user@example.com",
                "password": "stateful-password-secret",
            }),
            &mut state,
        )
        .await,
    );
    let request_count_before = document_requests.load(Ordering::SeqCst);

    let login = execute_command(
        &json!({ "id": "6", "action": "auth_login", "name": profile_name.clone(), "noNavigate": true }),
        &mut state,
    )
    .await;
    assert_success(&login);
    assert_eq!(get_data(&login)["loggedIn"], true);
    assert_eq!(
        document_requests.load(Ordering::SeqCst),
        request_count_before
    );

    let verify = execute_command(
        &json!({
            "id": "7",
            "action": "evaluate",
            "script": "({ marker: window.__marker, revealed: !!window.__revealed, submitted: !!window.__submitted, user: document.querySelector('input[type=email]').value, pass: document.querySelector('input[type=password]').value })",
        }),
        &mut state,
    )
    .await;
    assert_success(&verify);
    let result = &get_data(&verify)["result"];
    assert_eq!(result["marker"], "preserved");
    assert_eq!(result["revealed"], true);
    assert_eq!(result["submitted"], true);
    assert_eq!(result["user"], "stateful-user@example.com");
    assert_eq!(result["pass"], "stateful-password-secret");

    let _ = execute_command(
        &json!({ "id": "8", "action": "auth_delete", "name": profile_name }),
        &mut state,
    )
    .await;
    assert_success(&execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await);
}

#[cfg(unix)]
#[tokio::test]
#[ignore]
async fn e2e_auth_login_no_navigate_preserves_state_with_credential_provider() {
    use std::os::unix::fs::PermissionsExt;

    let (base_url, document_requests, _server) = start_stateful_auth_login_server().await;
    let plugin_dir = tempfile::tempdir().unwrap();
    let plugin_path = plugin_dir.path().join("stateful-credential-provider");
    std::fs::write(
        &plugin_path,
        format!(
            r#"#!/bin/sh
cat >/dev/null
printf '%s' '{{"protocol":"agent-browser.plugin.v1","success":true,"credential":{{"username":"provider-user@example.com","password":"provider-password-secret","url":"{}/provider/login"}}}}'
"#,
            base_url
        ),
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&plugin_path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&plugin_path, permissions).unwrap();

    let mut state = DaemonState::new();
    assert_success(
        &execute_command(
            &json!({ "id": "1", "action": "launch", "headless": true }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "2", "action": "navigate", "url": format!("{}/flow/provider", base_url) }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "3", "action": "click", "selector": "#reveal" }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "4", "action": "evaluate", "script": "window.__marker = 'provider-preserved'" }),
            &mut state,
        )
        .await,
    );
    let request_count_before = document_requests.load(Ordering::SeqCst);

    let login = execute_command(
        &json!({
            "id": "5",
            "action": "auth_login",
            "name": "provider-stateful-login",
            "noNavigate": true,
            "credentialProvider": "mock",
            "plugins": [{
                "name": "mock",
                "command": plugin_path.to_string_lossy(),
                "capabilities": ["credential.read"]
            }]
        }),
        &mut state,
    )
    .await;
    assert_success(&login);
    let serialized_login = serde_json::to_string(&login).unwrap();
    assert!(!serialized_login.contains("provider-user@example.com"));
    assert!(!serialized_login.contains("provider-password-secret"));
    assert_eq!(
        document_requests.load(Ordering::SeqCst),
        request_count_before
    );

    let verify = execute_command(
        &json!({
            "id": "6",
            "action": "evaluate",
            "script": "({ marker: window.__marker, submitted: !!window.__submitted, user: document.querySelector('input[type=email]').value, pass: document.querySelector('input[type=password]').value })",
        }),
        &mut state,
    )
    .await;
    assert_success(&verify);
    let result = &get_data(&verify)["result"];
    assert_eq!(result["marker"], "provider-preserved");
    assert_eq!(result["submitted"], true);
    assert_eq!(result["user"], "provider-user@example.com");
    assert_eq!(result["pass"], "provider-password-secret");

    assert_success(&execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await);
}

#[tokio::test]
#[ignore]
async fn e2e_auth_login_rejects_cross_origin_no_navigate() {
    let (active_origin, _, _active_server) = start_stateful_auth_login_server().await;
    let (credential_origin, credential_requests, _credential_server) =
        start_stateful_auth_login_server().await;
    let mut state = DaemonState::new();
    let profile_name = unique_auth_profile_name("origin-mismatch");

    assert_success(
        &execute_command(
            &json!({ "id": "1", "action": "launch", "headless": true }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "2", "action": "navigate", "url": format!("{}/flow/current-path-sentinel", active_origin) }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({ "id": "3", "action": "click", "selector": "#reveal" }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({
                "id": "4",
                "action": "auth_save",
                "name": profile_name.clone(),
                "url": format!("{}/credential-path-sentinel?credential-query-sentinel#credential-fragment-sentinel", credential_origin),
                "username": "username-secret-sentinel",
                "password": "password-secret-sentinel",
            }),
            &mut state,
        )
        .await,
    );

    let login = execute_command(
        &json!({ "id": "5", "action": "auth_login", "name": profile_name.clone(), "noNavigate": true }),
        &mut state,
    )
    .await;
    assert_eq!(login["success"], false);
    let error = login["error"].as_str().unwrap_or_default();
    assert!(error.contains("does not match credential origin"));
    for sentinel in [
        "current-path-sentinel",
        "credential-path-sentinel",
        "credential-query-sentinel",
        "credential-fragment-sentinel",
        "username-secret-sentinel",
        "password-secret-sentinel",
    ] {
        assert!(!error.contains(sentinel));
    }
    assert_eq!(credential_requests.load(Ordering::SeqCst), 0);

    let verify = execute_command(
        &json!({
            "id": "6",
            "action": "evaluate",
            "script": "({ user: document.querySelector('input[type=email]').value, pass: document.querySelector('input[type=password]').value, submitted: !!window.__submitted })",
        }),
        &mut state,
    )
    .await;
    assert_success(&verify);
    let result = &get_data(&verify)["result"];
    assert_eq!(result["user"], "");
    assert_eq!(result["pass"], "");
    assert_eq!(result["submitted"], false);

    let _ = execute_command(
        &json!({ "id": "7", "action": "auth_delete", "name": profile_name }),
        &mut state,
    )
    .await;
    assert_success(&execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await);
}

#[tokio::test]
#[ignore]
async fn e2e_auth_login_no_navigate_requires_usable_active_page() {
    let mut state = DaemonState::new();
    let profile_name = unique_auth_profile_name("missing-page");

    assert_success(
        &execute_command(
            &json!({
                "id": "1",
                "action": "auth_save",
                "name": profile_name.clone(),
                "url": "https://example.com/login",
                "username": "unused-user",
                "password": "unused-password",
            }),
            &mut state,
        )
        .await,
    );
    let login = execute_command(
        &json!({ "id": "2", "action": "auth_login", "name": profile_name.clone(), "noNavigate": true }),
        &mut state,
    )
    .await;
    assert_eq!(login["success"], false);
    assert!(login["error"]
        .as_str()
        .unwrap_or_default()
        .contains("requires an existing active HTTP(S) browser page"));
    assert!(
        state.browser.is_none(),
        "no-navigation login must not launch a browser"
    );

    let _ = execute_command(
        &json!({ "id": "3", "action": "auth_delete", "name": profile_name }),
        &mut state,
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn e2e_auth_login_waits_for_delayed_spa_form_render() {
    let (base_url, _server) = start_delayed_login_server(800, 100).await;
    let mut state = DaemonState::new();

    let profile_name = format!(
        "e2e-auth-login-spa-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_millis()
    );

    let launch = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&launch);

    let save = execute_command(
        &json!({
            "id": "2",
            "action": "auth_save",
            "name": profile_name.clone(),
            "url": format!("{}/login", base_url),
            "username": "user@example.com",
            "password": "super-secret",
        }),
        &mut state,
    )
    .await;
    assert_success(&save);

    let login = execute_command(
        &json!({ "id": "3", "action": "auth_login", "name": profile_name.clone() }),
        &mut state,
    )
    .await;
    assert_success(&login);
    assert_eq!(get_data(&login)["loggedIn"], true);

    let current_url = execute_command(&json!({ "id": "3b", "action": "url" }), &mut state).await;
    assert_success(&current_url);
    assert!(get_data(&current_url)["url"]
        .as_str()
        .unwrap_or_default()
        .starts_with(&format!("{}/login", base_url)));

    let verify = execute_command(
        &json!({
            "id": "4",
            "action": "evaluate",
            "script": "({ user: document.querySelector('input[type=email]')?.value ?? '', pass: document.querySelector('input[type=password]')?.value ?? '', search: document.querySelector('#search')?.value ?? '', submitted: !!window.__submitted })",
        }),
        &mut state,
    )
    .await;
    assert_success(&verify);
    let result = &get_data(&verify)["result"];
    assert_eq!(result["user"], "user@example.com");
    assert_eq!(result["pass"], "super-secret");
    assert_eq!(result["search"], "");
    assert_eq!(result["submitted"], true);

    let _ = execute_command(
        &json!({ "id": "5", "action": "auth_delete", "name": profile_name }),
        &mut state,
    )
    .await;

    let close = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&close);
}

// ---------------------------------------------------------------------------
// Origin-scoped --headers tests
// ---------------------------------------------------------------------------

/// Headers passed via --headers on open persist for subsequent same-origin
/// navigations (the core regression from the Rust rewrite).
#[tokio::test]
#[ignore]
async fn e2e_headers_persist_same_origin_navigation() {
    let (base_url, _server) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate with --headers.
    let resp = execute_command(
        &json!({
            "id": "2", "action": "navigate",
            "url": format!("{}/first", base_url),
            "headers": { "X-Test": "scoped" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate to the same origin WITHOUT --headers.
    let resp = execute_command(
        &json!({
            "id": "3", "action": "navigate",
            "url": format!("{}/second", base_url),
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // The page body is the echo JSON. Read it via evaluate.
    let resp = execute_command(
        &json!({
            "id": "4", "action": "evaluate",
            "script": "JSON.parse(document.body.innerText)",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let result = &get_data(&resp)["result"];
    assert_eq!(
        result["headers"]["X-Test"], "scoped",
        "X-Test header should persist on same-origin navigation without --headers"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Headers passed via --headers on open persist for in-page fetch/XHR to
/// the same origin.
#[tokio::test]
#[ignore]
async fn e2e_headers_persist_same_origin_fetch() {
    let (base_url, _server) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate with --headers.
    let resp = execute_command(
        &json!({
            "id": "2", "action": "navigate",
            "url": format!("{}/page", base_url),
            "headers": { "X-Test": "fetched" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // In-page fetch to the same origin (relative URL).
    let resp = execute_command(
        &json!({
            "id": "3", "action": "evaluate",
            "script": "fetch('/echo').then(r => r.json())",
            "await": true,
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let result = &get_data(&resp)["result"];
    assert_eq!(
        result["headers"]["X-Test"], "fetched",
        "X-Test header should be present on in-page fetch to same origin"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Headers set via --headers do NOT leak to a different origin.
#[tokio::test]
#[ignore]
async fn e2e_headers_do_not_leak_cross_origin() {
    let (server_a, _ha) = start_echo_server().await;
    let (server_b, _hb) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate to server A with --headers.
    let resp = execute_command(
        &json!({
            "id": "2", "action": "navigate",
            "url": format!("{}/page", server_a),
            "headers": { "X-Secret": "a-only" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate to server B (different origin) without --headers.
    let resp = execute_command(
        &json!({
            "id": "3", "action": "navigate",
            "url": format!("{}/page", server_b),
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "4", "action": "evaluate",
            "script": "JSON.parse(document.body.innerText)",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let result = &get_data(&resp)["result"];
    assert!(
        result["headers"].get("X-Secret").is_none(),
        "X-Secret header must NOT leak to a different origin, got: {}",
        result["headers"],
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// In-page fetch to a cross-origin URL must NOT include the origin-scoped
/// headers (sub-resource isolation).
#[tokio::test]
#[ignore]
async fn e2e_headers_do_not_leak_cross_origin_fetch() {
    let (server_a, _ha) = start_echo_server().await;
    let (server_b, _hb) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate to server A with --headers.
    let resp = execute_command(
        &json!({
            "id": "2", "action": "navigate",
            "url": format!("{}/page", server_a),
            "headers": { "X-Secret": "a-only" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Fetch from the page to server B (cross-origin sub-resource).
    let resp = execute_command(
        &json!({
            "id": "3", "action": "evaluate",
            "script": format!("fetch('{}/echo').then(r => r.json())", server_b),
            "await": true,
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let result = &get_data(&resp)["result"];
    assert!(
        result["headers"].get("X-Secret").is_none(),
        "X-Secret header must NOT leak to cross-origin fetch, got: {}",
        result["headers"],
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// `set headers` (global headers via the headers action) must not be
/// regressed — they should persist across navigations without being
/// cleared by the origin-scoped header logic.
#[tokio::test]
#[ignore]
async fn e2e_set_headers_not_regressed() {
    let (base_url, _server) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Set global headers via the `headers` action (not --headers on navigate).
    let resp = execute_command(
        &json!({
            "id": "2", "action": "headers",
            "headers": { "X-Global": "everywhere" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate — global headers should be present.
    let resp = execute_command(
        &json!({
            "id": "3", "action": "navigate",
            "url": format!("{}/page", base_url),
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "4", "action": "evaluate",
            "script": "JSON.parse(document.body.innerText)",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let result = &get_data(&resp)["result"];
    assert_eq!(
        result["headers"]["X-Global"], "everywhere",
        "Global headers set via `set headers` must persist across navigations"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Multiple origins each get their own independent headers.
#[tokio::test]
#[ignore]
async fn e2e_headers_multiple_origins_independent() {
    let (server_a, _ha) = start_echo_server().await;
    let (server_b, _hb) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Set headers for origin A.
    let resp = execute_command(
        &json!({
            "id": "2", "action": "navigate",
            "url": format!("{}/page", server_a),
            "headers": { "X-From": "alpha" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Set different headers for origin B.
    let resp = execute_command(
        &json!({
            "id": "3", "action": "navigate",
            "url": format!("{}/page", server_b),
            "headers": { "X-From": "beta" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Verify B got its own header.
    let resp = execute_command(
        &json!({ "id": "4", "action": "evaluate", "script": "JSON.parse(document.body.innerText)" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"]["headers"]["X-From"], "beta");

    // Navigate back to A — should get A's header, not B's.
    let resp = execute_command(
        &json!({ "id": "5", "action": "navigate", "url": format!("{}/check", server_a) }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "6", "action": "evaluate", "script": "JSON.parse(document.body.innerText)" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"]["headers"]["X-From"], "alpha");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Headers persist when navigating away to a different origin and back.
#[tokio::test]
#[ignore]
async fn e2e_headers_persist_after_roundtrip() {
    let (server_a, _ha) = start_echo_server().await;
    let (server_b, _hb) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Set headers for origin A.
    let resp = execute_command(
        &json!({
            "id": "2", "action": "navigate",
            "url": format!("{}/page", server_a),
            "headers": { "X-Persist": "roundtrip" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate away to B (no headers).
    let resp = execute_command(
        &json!({ "id": "3", "action": "navigate", "url": format!("{}/page", server_b) }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Navigate back to A without --headers.
    let resp = execute_command(
        &json!({ "id": "4", "action": "navigate", "url": format!("{}/back", server_a) }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "5", "action": "evaluate", "script": "JSON.parse(document.body.innerText)" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["result"]["headers"]["X-Persist"],
        "roundtrip",
        "Headers should persist after navigating away and back to the same origin"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Passing --headers a second time to the same origin replaces the previous headers.
#[tokio::test]
#[ignore]
async fn e2e_headers_override_same_origin() {
    let (base_url, _server) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Set initial headers.
    let resp = execute_command(
        &json!({
            "id": "2", "action": "navigate",
            "url": format!("{}/first", base_url),
            "headers": { "X-Version": "v1" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Override with new headers.
    let resp = execute_command(
        &json!({
            "id": "3", "action": "navigate",
            "url": format!("{}/second", base_url),
            "headers": { "X-Version": "v2" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "4", "action": "evaluate", "script": "JSON.parse(document.body.innerText)" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["result"]["headers"]["X-Version"],
        "v2",
        "Second --headers should replace the first for the same origin"
    );

    // Subsequent navigation without --headers should use v2.
    let resp = execute_command(
        &json!({ "id": "5", "action": "navigate", "url": format!("{}/third", base_url) }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "6", "action": "evaluate", "script": "JSON.parse(document.body.innerText)" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"]["headers"]["X-Version"], "v2");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// `set headers` (global) and `--headers` (origin-scoped) stack together.
#[tokio::test]
#[ignore]
async fn e2e_global_and_scoped_headers_stack() {
    let (base_url, _server) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Set global headers via `set headers`.
    let resp = execute_command(
        &json!({
            "id": "2", "action": "headers",
            "headers": { "X-Global": "everywhere" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Set origin-scoped headers via --headers.
    let resp = execute_command(
        &json!({
            "id": "3", "action": "navigate",
            "url": format!("{}/page", base_url),
            "headers": { "X-Scoped": "this-origin" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "4", "action": "evaluate", "script": "JSON.parse(document.body.innerText)" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let headers = &get_data(&resp)["result"]["headers"];
    assert_eq!(
        headers["X-Global"], "everywhere",
        "Global header should be present alongside scoped header"
    );
    assert_eq!(
        headers["X-Scoped"], "this-origin",
        "Scoped header should be present alongside global header"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Origin-scoped headers with different casing than the browser's original
/// request headers must not produce duplicates (HTTP headers are
/// case-insensitive per RFC 7230).
#[tokio::test]
#[ignore]
async fn e2e_headers_case_insensitive_no_duplicates() {
    let (base_url, _server) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Chrome sends "Accept: ..." by default on navigations. Pass "accept"
    // (lowercase) via --headers to verify the merge is case-insensitive
    // and doesn't produce a duplicate Accept header.
    let resp = execute_command(
        &json!({
            "id": "2", "action": "navigate",
            "url": format!("{}/page", base_url),
            "headers": { "accept": "application/test" },
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "3", "action": "evaluate",
            "script": "JSON.parse(document.body.innerText)",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let result = &get_data(&resp)["result"]["headers"];

    // The echo server stores headers keyed by name as received on the wire.
    // If deduplication works, only our custom "accept" value should appear
    // (Chrome's original "Accept: text/html,..." should be suppressed).
    let accept_val = result
        .get("accept")
        .or_else(|| result.get("Accept"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert_eq!(
        accept_val, "application/test",
        "Case-insensitive merge should replace Chrome's Accept header, got headers: {}",
        result,
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Regression: externally opened tabs must appear in tab_list (#1037)
//
// When connected to Chrome (launched or via --cdp), a tab opened outside of
// agent-browser (e.g. by the user or another CDP client) should be detected
// and listed. Previously, chrome://newtab/ was filtered by
// is_internal_chrome_target, and Target.targetInfoChanged for untracked
// targets was silently ignored.
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_externally_opened_tab_detected() {
    let mut state = DaemonState::new();

    // Launch headless Chrome
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Verify initial tab count
    let resp = execute_command(&json!({ "id": "2", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let initial_count = get_data(&resp)["tabs"].as_array().unwrap().len();

    // Simulate an external client opening a new tab via the browser-level CDP
    // session (no sessionId). This mirrors what happens when a user manually
    // opens a tab while agent-browser is connected via --cdp.
    let browser = state.browser.as_ref().expect("browser should be launched");
    let _: Value = browser
        .client
        .send_command(
            "Target.createTarget",
            Some(json!({ "url": "data:text/html,<h1>External Tab</h1>" })),
            None, // browser-level session
        )
        .await
        .expect("Target.createTarget should succeed");

    // Give Chrome a moment to fire targetCreated / targetInfoChanged events
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Drain events by issuing tab_list — this triggers execute_command's
    // drain_cdp_events path which processes new and changed targets.
    let resp = execute_command(&json!({ "id": "3", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap();

    assert_eq!(
        tabs.len(),
        initial_count + 1,
        "Externally opened tab should appear in tab_list, got: {:?}",
        tabs,
    );

    // Verify the new tab's URL is the data URL we navigated to
    let new_tab = tabs.iter().find(|t| {
        t["url"]
            .as_str()
            .is_some_and(|u| u.starts_with("data:text/html"))
    });
    assert!(
        new_tab.is_some(),
        "Should find the externally opened tab by URL, tabs: {:?}",
        tabs,
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Regression: issue #993 — launch options change must trigger relaunch
// ---------------------------------------------------------------------------

/// When the browser is already running and a second launch command arrives with
/// different options (e.g., extensions added), the daemon must relaunch the
/// browser instead of silently reusing the old one.
///
/// Before the fix, `handle_launch` only checked connection type and liveness,
/// so changed options like extensions were ignored and the old browser was reused.
#[tokio::test]
#[ignore]
async fn e2e_relaunch_on_options_change() {
    let mut state = DaemonState::new();

    // First launch — headless, no extensions.
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["launched"], true);
    assert!(
        get_data(&resp).get("reused").is_none(),
        "first launch must not be a reuse"
    );

    // Second launch — same options → should reuse.
    let resp = execute_command(
        &json!({ "id": "2", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["reused"],
        true,
        "identical options must reuse the browser"
    );

    // Third launch — different options (userAgent changed) → must relaunch, not reuse.
    // We use userAgent instead of extensions because extensions force headed mode,
    // which requires a display server and fails in headless CI environments.
    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "launch",
            "headless": true,
            "userAgent": "agent-browser-test/1.0"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(
        get_data(&resp).get("reused").is_none(),
        "changed options must trigger a relaunch, not reuse (issue #993)"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Stream: URL events follow active main-frame navigation
// ---------------------------------------------------------------------------

async fn start_stream_navigation_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("stream navigation server should bind");
    let port = listener
        .local_addr()
        .expect("stream navigation server should have an address")
        .port();
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut buffer = vec![0u8; 4096];
                let Ok(size) = stream.read(&mut buffer).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buffer[..size]);
                let request_target = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                let path = request_target.split('?').next().unwrap_or("/");
                if let Some(destination) = path.strip_prefix("/redirect/") {
                    let location = format!("/landed/{destination}");
                    let response = format!(
                        "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    return;
                }
                let body = match path {
                    "/child" => {
                        "<!doctype html><title>child</title><p id=\"child\">child</p>"
                    }
                    _ => {
                        "<!doctype html><title>main</title><a id=\"anchor\" href=\"#section\">anchor</a><div id=\"section\">section</div><iframe id=\"child\" src=\"/child\"></iframe>"
                    }
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    });
    (format!("http://127.0.0.1:{port}"), handle)
}

async fn next_stream_url(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> String {
    loop {
        let message = tokio::time::timeout(tokio::time::Duration::from_secs(5), ws.next())
            .await
            .expect("stream should emit a URL message")
            .expect("stream should stay open")
            .expect("stream message should be valid");
        if !message.is_text() {
            continue;
        }
        let payload: Value =
            serde_json::from_str(message.to_text().expect("message should be text"))
                .expect("stream payload should be JSON");
        if payload["type"] == "url" {
            return payload["url"]
                .as_str()
                .expect("URL message should carry a URL")
                .to_string();
        }
    }
}

async fn wait_for_stream_navigation_ready(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) {
    loop {
        let message = tokio::time::timeout(tokio::time::Duration::from_secs(10), ws.next())
            .await
            .expect("stream navigation observer should become ready")
            .expect("stream should stay open")
            .expect("stream message should be valid");
        if !message.is_text() {
            continue;
        }
        let payload: Value =
            serde_json::from_str(message.to_text().expect("message should be text"))
                .expect("stream payload should be JSON");
        if payload["type"] == "status"
            && payload["connected"] == true
            && payload["screencasting"] == true
        {
            return;
        }
    }
}

async fn expect_no_stream_url(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    duration: tokio::time::Duration,
) {
    let deadline = tokio::time::Instant::now() + duration;
    while tokio::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let result = tokio::time::timeout(remaining, ws.next()).await;
        let Ok(Some(Ok(message))) = result else {
            return;
        };
        if !message.is_text() {
            continue;
        }
        let payload: Value =
            serde_json::from_str(message.to_text().expect("message should be text"))
                .expect("stream payload should be JSON");
        assert_ne!(
            payload["type"], "url",
            "unexpected background or child URL: {payload}"
        );
    }
}

async fn next_collected_stream_message(
    messages: &mut tokio::sync::mpsc::UnboundedReceiver<Value>,
    iteration: usize,
    phase: &str,
) -> Value {
    tokio::time::timeout(tokio::time::Duration::from_secs(5), messages.recv())
        .await
        .unwrap_or_else(|_| panic!("stress iteration {iteration} timed out during {phase}"))
        .unwrap_or_else(|| {
            panic!("stress iteration {iteration} stream collector stopped during {phase}")
        })
}

async fn wait_for_active_stream_tab(
    messages: &mut tokio::sync::mpsc::UnboundedReceiver<Value>,
    tab_id: &str,
    iteration: usize,
) {
    loop {
        let payload =
            next_collected_stream_message(messages, iteration, "active-tab stream rebind").await;
        if payload["type"] != "tabs" {
            continue;
        }
        let is_active = payload["tabs"].as_array().is_some_and(|tabs| {
            tabs.iter()
                .any(|tab| tab["tabId"] == tab_id && tab["active"].as_bool() == Some(true))
        });
        if is_active {
            return;
        }
    }
}

async fn wait_for_stress_stream_url(
    messages: &mut tokio::sync::mpsc::UnboundedReceiver<Value>,
    expected_url: &str,
    forbidden_marker: &str,
    iteration: usize,
) {
    loop {
        let payload =
            next_collected_stream_message(messages, iteration, "active URL convergence").await;
        if payload["type"] != "url" {
            continue;
        }
        let url = payload["url"]
            .as_str()
            .expect("stress URL payload should contain a URL");
        assert!(
            !url.contains(forbidden_marker),
            "stress iteration {iteration} attributed the previous tab URL to the new active tab: {payload}"
        );
        if url == expected_url {
            return;
        }
    }
}

async fn expect_no_forbidden_stream_url(
    messages: &mut tokio::sync::mpsc::UnboundedReceiver<Value>,
    forbidden_marker: &str,
    iteration: usize,
) {
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_millis(100);
    while tokio::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let Ok(Some(payload)) = tokio::time::timeout(remaining, messages.recv()).await else {
            return;
        };
        if payload["type"] != "url" {
            continue;
        }
        let url = payload["url"]
            .as_str()
            .expect("stress URL payload should contain a URL");
        assert!(
            !url.contains(forbidden_marker),
            "stress iteration {iteration} emitted a forbidden URL after convergence: {payload}"
        );
    }
}

async fn seeded_stream_tabs(port: u64, iteration: usize) -> Vec<Value> {
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .expect("stress probe should connect to runtime stream");
    loop {
        let message = tokio::time::timeout(tokio::time::Duration::from_secs(5), ws.next())
            .await
            .unwrap_or_else(|_| {
                panic!("stress iteration {iteration} probe timed out waiting for tabs")
            })
            .expect("stress probe stream should stay open")
            .expect("stress probe message should be valid");
        if !message.is_text() {
            continue;
        }
        let payload: Value =
            serde_json::from_str(message.to_text().expect("probe message should be text"))
                .expect("probe stream payload should be JSON");
        if payload["type"] == "tabs" {
            return payload["tabs"]
                .as_array()
                .expect("probe tabs payload should be an array")
                .clone();
        }
    }
}

#[tokio::test]
#[ignore]
async fn e2e_stream_url_tracks_active_main_frame_navigation_categories() {
    let guard = EnvGuard::new(&["AGENT_BROWSER_SOCKET_DIR", "AGENT_BROWSER_SESSION"]);
    let socket_dir = std::env::temp_dir().join(format!(
        "agent-browser-e2e-stream-url-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&socket_dir).expect("socket dir should be created");
    guard.set(
        "AGENT_BROWSER_SOCKET_DIR",
        socket_dir.to_str().expect("socket dir should be utf-8"),
    );
    guard.set("AGENT_BROWSER_SESSION", "e2e-stream-url");

    let (base_url, server) = start_stream_navigation_server().await;
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "stream_enable", "port": 0 }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let port = get_data(&resp)["port"]
        .as_u64()
        .expect("stream enable should report the bound port");

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": format!("{base_url}/") }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .expect("websocket client should connect to runtime stream");
    wait_for_stream_navigation_ready(&mut ws).await;

    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "evaluate",
            "script": "history.pushState({}, '', '/spa'); location.href"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(next_stream_url(&mut ws).await, format!("{base_url}/spa"));

    let resp = execute_command(
        &json!({
            "id": "4",
            "action": "evaluate",
            "script": "location.hash = 'section'; location.href"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        next_stream_url(&mut ws).await,
        format!("{base_url}/spa#section")
    );

    let resp = execute_command(
        &json!({
            "id": "5",
            "action": "evaluate",
            "script": "document.querySelector('#child').contentWindow.history.pushState({}, '', '/child-spa'); 'done'"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    expect_no_stream_url(&mut ws, tokio::time::Duration::from_millis(500)).await;

    let resp = execute_command(
        &json!({ "id": "6", "action": "navigate", "url": format!("{base_url}/full") }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(next_stream_url(&mut ws).await, format!("{base_url}/full"));

    let resp = execute_command(
        &json!({
            "id": "7",
            "action": "tab_new",
            "url": format!("{base_url}/")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let background_session = state
        .browser
        .as_ref()
        .expect("browser should exist")
        .pages_list()
        .into_iter()
        .find(|page| page.tab_id == 2)
        .expect("background tab should exist")
        .session_id;
    let resp = execute_command(
        &json!({ "id": "8", "action": "tab_switch", "tabId": "t1" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    while tokio::time::timeout(tokio::time::Duration::from_millis(100), ws.next())
        .await
        .is_ok()
    {}

    let resp = execute_command(
        &json!({
            "id": "9",
            "action": "evaluate",
            "script": "history.pushState({}, '', '/after-switch'); location.href"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        next_stream_url(&mut ws).await,
        format!("{base_url}/after-switch")
    );

    state
        .browser
        .as_ref()
        .expect("browser should exist")
        .client
        .send_command(
            "Page.navigate",
            Some(json!({ "url": format!("{base_url}/background-full") })),
            Some(&background_session),
        )
        .await
        .expect("background tab should navigate");
    expect_no_stream_url(&mut ws, tokio::time::Duration::from_millis(500)).await;

    state
        .browser
        .as_ref()
        .expect("browser should exist")
        .client
        .send_command(
            "Target.createTarget",
            Some(json!({ "url": format!("{base_url}/external") })),
            None,
        )
        .await
        .expect("external tab should open");
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    let resp = execute_command(&json!({ "id": "10", "action": "url" }), &mut state).await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["url"], format!("{base_url}/external"));

    let external_session = state
        .browser
        .as_ref()
        .expect("browser should exist")
        .active_session_id()
        .expect("external tab should become active")
        .to_string();
    state
        .browser
        .as_ref()
        .expect("browser should exist")
        .client
        .send_command(
            "Runtime.evaluate",
            Some(json!({
                "expression": "history.pushState({}, '', '/external-spa'); location.href"
            })),
            Some(&external_session),
        )
        .await
        .expect("external tab should navigate within its document");
    assert_eq!(
        next_stream_url(&mut ws).await,
        format!("{base_url}/external-spa")
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    server.abort();
    let _ = std::fs::remove_dir_all(&socket_dir);
}

#[tokio::test]
#[ignore]
async fn e2e_lightpanda_stream_url_tracks_active_full_navigation() {
    let lightpanda_bin = match std::env::var("LIGHTPANDA_BIN") {
        Ok(path) if !path.is_empty() => path,
        _ => return,
    };
    let guard = EnvGuard::new(&["AGENT_BROWSER_SOCKET_DIR", "AGENT_BROWSER_SESSION"]);
    let socket_dir = std::env::temp_dir().join(format!(
        "agent-browser-e2e-lightpanda-stream-url-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&socket_dir).expect("socket dir should be created");
    guard.set(
        "AGENT_BROWSER_SOCKET_DIR",
        socket_dir.to_str().expect("socket dir should be utf-8"),
    );
    guard.set("AGENT_BROWSER_SESSION", "e2e-lightpanda-stream-url");

    let (base_url, server) = start_stream_navigation_server().await;
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "lp-stream", "action": "stream_enable", "port": 0 }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let port = get_data(&resp)["port"]
        .as_u64()
        .expect("stream enable should report the bound port");

    let resp = tokio::time::timeout(
        tokio::time::Duration::from_secs(20),
        execute_command(
            &json!({
                "id": "lp-launch",
                "action": "launch",
                "headless": true,
                "engine": "lightpanda",
                "executablePath": lightpanda_bin
            }),
            &mut state,
        ),
    )
    .await
    .expect("Lightpanda stream launch should not hang");
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "lp-initial",
            "action": "navigate",
            "url": format!("{base_url}/lp-initial")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .expect("Lightpanda stream client should connect");
    while tokio::time::timeout(tokio::time::Duration::from_millis(100), ws.next())
        .await
        .is_ok()
    {}

    let active_before_background = format!("{base_url}/lp-active-before-background");
    let resp = execute_command(
        &json!({
            "id": "lp-active-before",
            "action": "navigate",
            "url": active_before_background
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(next_stream_url(&mut ws).await, active_before_background);

    let resp = execute_command(
        &json!({
            "id": "lp-child-navigation",
            "action": "evaluate",
            "script": "document.querySelector('#child').src = '/lp-forbidden-child'; 'done'"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    expect_no_stream_url(&mut ws, tokio::time::Duration::from_millis(500)).await;

    let tabs = seeded_stream_tabs(port, 0).await;
    let active = tabs
        .iter()
        .find(|tab| tab["active"].as_bool() == Some(true))
        .expect("Lightpanda reconnect should seed an active tab");
    assert_eq!(active["tabId"], "t1");
    assert_eq!(active["url"], active_before_background);

    let final_url = format!("{base_url}/lp-active-final");
    let resp = execute_command(
        &json!({
            "id": "lp-active-final",
            "action": "navigate",
            "url": final_url
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(next_stream_url(&mut ws).await, final_url);

    let resp = execute_command(&json!({ "id": "lp-close", "action": "close" }), &mut state).await;
    assert_success(&resp);
    server.abort();
    let _ = std::fs::remove_dir_all(&socket_dir);
}

#[tokio::test]
#[ignore]
async fn e2e_stream_url_survives_real_world_navigation_stress() {
    let guard = EnvGuard::new(&["AGENT_BROWSER_SOCKET_DIR", "AGENT_BROWSER_SESSION"]);
    let socket_dir = std::env::temp_dir().join(format!(
        "agent-browser-e2e-stream-url-stress-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&socket_dir).expect("socket dir should be created");
    guard.set(
        "AGENT_BROWSER_SOCKET_DIR",
        socket_dir.to_str().expect("socket dir should be utf-8"),
    );
    guard.set("AGENT_BROWSER_SESSION", "e2e-stream-url-stress");

    let iterations = std::env::var("AGENT_BROWSER_STRESS_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(40);
    let mut seed = std::env::var("AGENT_BROWSER_STRESS_SEED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1677);

    let (base_url, server) = start_stream_navigation_server().await;
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "stress-stream", "action": "stream_enable", "port": 0 }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let port = get_data(&resp)["port"]
        .as_u64()
        .expect("stream enable should report the bound port");

    let resp = execute_command(
        &json!({
            "id": "stress-tab-a",
            "action": "navigate",
            "url": format!("{base_url}/app-a")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({
            "id": "stress-tab-b",
            "action": "tab_new",
            "url": format!("{base_url}/app-b")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let pages = state
        .browser
        .as_ref()
        .expect("stress browser should exist")
        .pages_list();
    let session_a = pages
        .iter()
        .find(|page| page.tab_id == 1)
        .expect("stress tab A should exist")
        .session_id
        .clone();
    let session_b = pages
        .iter()
        .find(|page| page.tab_id == 2)
        .expect("stress tab B should exist")
        .session_id
        .clone();
    let client = Arc::clone(
        &state
            .browser
            .as_ref()
            .expect("stress browser should exist")
            .client,
    );

    let (fast_ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .expect("stress collector should connect to runtime stream");
    let (message_tx, mut messages) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let collector = tokio::spawn(async move {
        let mut ws = fast_ws;
        while let Some(Ok(message)) = ws.next().await {
            if !message.is_text() {
                continue;
            }
            let Ok(payload) = serde_json::from_str::<Value>(
                message.to_text().expect("collector message should be text"),
            ) else {
                continue;
            };
            if message_tx.send(payload).is_err() {
                break;
            }
        }
    });
    wait_for_active_stream_tab(&mut messages, "t2", 0).await;

    let (mut slow_ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .expect("slow stress client should connect to runtime stream");

    let mut active_tab = "t2";
    for iteration in 0..iterations {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let delay_ms = 1 + (seed % 12);
        let target_tab = if active_tab == "t1" { "t2" } else { "t1" };
        let old_session = if active_tab == "t1" {
            session_a.clone()
        } else {
            session_b.clone()
        };
        let forbidden_marker = format!("forbidden-{iteration}");
        let same_document_path = format!("/{forbidden_marker}-old-spa");
        let redirect_url = format!("{base_url}/redirect/{forbidden_marker}-old-full");

        let late_client = Arc::clone(&client);
        let late_session = old_session.clone();
        let late_same_document_path = same_document_path.clone();
        let late_child_marker = forbidden_marker.clone();
        let late_same_document = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
            let expression = format!(
                "history.pushState({{}}, '', {}); const child = document.querySelector('#child'); if (child?.contentWindow) child.contentWindow.history.pushState({{}}, '', '/{late_child_marker}-old-child'); location.href",
                serde_json::to_string(&late_same_document_path)
                    .expect("stress path should serialize")
            );
            late_client
                .send_command(
                    "Runtime.evaluate",
                    Some(json!({ "expression": expression })),
                    Some(&late_session),
                )
                .await
        });

        let redirect_client = Arc::clone(&client);
        let redirect_session = old_session.clone();
        let late_redirect = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms + 1)).await;
            redirect_client
                .send_command(
                    "Page.navigate",
                    Some(json!({ "url": redirect_url })),
                    Some(&redirect_session),
                )
                .await
        });

        let resp = execute_command(
            &json!({
                "id": format!("stress-switch-{iteration}"),
                "action": "tab_switch",
                "tabId": target_tab
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        wait_for_active_stream_tab(&mut messages, target_tab, iteration).await;
        tokio::time::timeout(tokio::time::Duration::from_secs(5), late_same_document)
            .await
            .unwrap_or_else(|_| panic!("stress iteration {iteration} late SPA task timed out"))
            .expect("late SPA task should join")
            .expect("late SPA CDP command should succeed");
        tokio::time::timeout(tokio::time::Duration::from_secs(5), late_redirect)
            .await
            .unwrap_or_else(|_| panic!("stress iteration {iteration} late redirect task timed out"))
            .expect("late redirect task should join")
            .expect("late redirect CDP command should succeed");

        let resp = execute_command(
            &json!({
                "id": format!("stress-child-{iteration}"),
                "action": "evaluate",
                "script": format!(
                    "const child = document.querySelector('#child'); if (!child?.contentWindow) throw new Error('missing stress iframe'); child.contentWindow.history.pushState({{}}, '', '/{forbidden_marker}-active-child'); 'done'"
                )
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let expected_url = format!("{base_url}/{target_tab}-active-{iteration}");
        let resp = execute_command(
            &json!({
                "id": format!("stress-active-{iteration}"),
                "action": "evaluate",
                "script": format!(
                    "history.replaceState({{}}, '', {}); location.href",
                    serde_json::to_string(&expected_url).expect("stress URL should serialize")
                )
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        wait_for_stress_stream_url(&mut messages, &expected_url, &forbidden_marker, iteration)
            .await;

        let resp = execute_command(
            &json!({ "id": format!("stress-url-{iteration}"), "action": "url" }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert_eq!(
            get_data(&resp)["url"],
            expected_url,
            "stress iteration {iteration} active browser URL diverged"
        );
        expect_no_forbidden_stream_url(&mut messages, &forbidden_marker, iteration).await;

        if iteration % 5 == 0 {
            let tabs = seeded_stream_tabs(port, iteration).await;
            let active = tabs
                .iter()
                .find(|tab| tab["active"].as_bool() == Some(true))
                .unwrap_or_else(|| {
                    panic!("stress iteration {iteration} reconnect had no active tab")
                });
            assert_eq!(
                active["tabId"], target_tab,
                "stress iteration {iteration} reconnect seeded the wrong active tab"
            );
            assert_eq!(
                active["url"], expected_url,
                "stress iteration {iteration} reconnect seeded a stale URL"
            );
        }

        if iteration % 7 == 6 {
            drop(slow_ws);
            let (replacement, _) =
                tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
                    .await
                    .expect("replacement slow stress client should connect");
            slow_ws = replacement;
        }

        active_tab = target_tab;
    }

    drop(slow_ws);
    collector.abort();
    let resp = execute_command(
        &json!({ "id": "stress-close", "action": "close" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    server.abort();
    let _ = std::fs::remove_dir_all(&socket_dir);
}

// ---------------------------------------------------------------------------
// Stream: custom viewport is reflected in screencast frame metadata
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_stream_frame_metadata_respects_custom_viewport() {
    let guard = EnvGuard::new(&["AGENT_BROWSER_SOCKET_DIR", "AGENT_BROWSER_SESSION"]);
    let socket_dir = std::env::temp_dir().join(format!(
        "agent-browser-e2e-stream-viewport-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&socket_dir).expect("socket dir should be created");
    guard.set(
        "AGENT_BROWSER_SOCKET_DIR",
        socket_dir.to_str().expect("socket dir should be utf-8"),
    );
    guard.set("AGENT_BROWSER_SESSION", "e2e-stream-viewport");

    let mut state = DaemonState::new();

    // Enable stream on an ephemeral port
    let resp = execute_command(
        &json!({ "id": "1", "action": "stream_enable", "port": 0 }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let port = get_data(&resp)["port"]
        .as_u64()
        .expect("stream enable should report the bound port");

    // Set a custom viewport before launching the browser
    let resp = execute_command(
        &json!({ "id": "2", "action": "viewport", "width": 800, "height": 600 }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Connect a WebSocket client
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .expect("websocket client should connect to runtime stream");

    // Navigate to trigger browser launch and screencast
    let resp = execute_command(
        &json!({ "id": "3", "action": "navigate", "url": "data:text/html,<h1>Viewport Test</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Wait for a frame whose JPEG dimensions match the custom viewport.
    // Early frames may arrive before Chrome fully applies the viewport resize,
    // so skip frames with stale dimensions rather than failing immediately.
    let mut found_frame = false;
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(15);
    while tokio::time::Instant::now() < deadline {
        let msg = tokio::time::timeout(tokio::time::Duration::from_secs(3), ws.next()).await;
        let Some(Ok(message)) = msg.ok().flatten() else {
            continue;
        };
        if !message.is_text() {
            continue;
        }
        let parsed: Value =
            serde_json::from_str(message.to_text().expect("text message should be readable"))
                .expect("stream payload should be valid JSON");
        if parsed.get("type") == Some(&json!("frame")) {
            let meta = &parsed["metadata"];
            assert_eq!(
                meta["deviceWidth"], 800,
                "frame metadata deviceWidth should match custom viewport, got: {}",
                meta
            );
            assert_eq!(
                meta["deviceHeight"], 600,
                "frame metadata deviceHeight should match custom viewport, got: {}",
                meta
            );

            let data_str = parsed
                .get("data")
                .and_then(|v| v.as_str())
                .expect("frame message should include base64-encoded 'data' field");
            use base64::Engine;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data_str)
                .expect("frame data should be valid base64");
            let (img_w, img_h) =
                jpeg_dimensions(&bytes).expect("frame data should be a valid JPEG with SOF marker");
            if img_w != 800 || img_h != 600 {
                continue;
            }

            found_frame = true;
            break;
        }
    }
    assert!(
        found_frame,
        "should have received a frame with JPEG dimensions 800x600 within the deadline"
    );

    // Cleanup
    let resp = execute_command(
        &json!({ "id": "4", "action": "stream_disable" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    let _ = std::fs::remove_dir_all(&socket_dir);
}

// ---------------------------------------------------------------------------
// Stream: a click is not queued behind a mouse sweep
// ---------------------------------------------------------------------------

/// Guards the no-await dispatch. Awaiting Chrome's reply per event serialized
/// the reader, so a click arrived one CDP round trip behind every mousemove
/// ahead of it: 300 moves delayed it by ~2.5s. Measured at ~6ms with the fix,
/// so the threshold has three orders of magnitude of headroom and fails only on
/// a real regression.
#[tokio::test]
#[ignore]
async fn e2e_stream_click_is_not_queued_behind_a_mouse_sweep() {
    let guard = EnvGuard::new(&["AGENT_BROWSER_SOCKET_DIR", "AGENT_BROWSER_SESSION"]);
    let socket_dir = std::env::temp_dir().join(format!(
        "agent-browser-e2e-stream-latency-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&socket_dir).expect("socket dir should be created");
    guard.set(
        "AGENT_BROWSER_SOCKET_DIR",
        socket_dir.to_str().expect("socket dir should be utf-8"),
    );
    guard.set("AGENT_BROWSER_SESSION", "e2e-stream-latency");

    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "stream_enable", "port": 0 }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let port = get_data(&resp)["port"]
        .as_u64()
        .expect("stream enable should report the bound port");

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<h1>latency</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Record the arrival time of the first mousedown inside the page.
    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "evaluate",
            "script": "window.__down = null; document.addEventListener('mousedown', () => { if (window.__down === null) window.__down = Date.now(); }); 'armed'"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    use futures_util::SinkExt;
    use tokio_tungstenite::tungstenite::Message;
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .expect("websocket client should connect to runtime stream");
    let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), ws.next()).await;

    // A sweep of moves, then the click that must not wait for them.
    for i in 0..300 {
        let mv = json!({
            "type": "input_mouse", "eventType": "mouseMoved",
            "x": 100 + (i % 400), "y": 100 + (i % 300),
            "button": "none", "clickCount": 0
        });
        ws.send(Message::Text(mv.to_string()))
            .await
            .expect("mouse move should be accepted by the stream socket");
    }
    let sent_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_millis() as u64;
    for event in ["mousePressed", "mouseReleased"] {
        let click = json!({
            "type": "input_mouse", "eventType": event,
            "x": 200, "y": 200, "button": "left", "clickCount": 1
        });
        ws.send(Message::Text(click.to_string()))
            .await
            .expect("click should be accepted by the stream socket");
    }

    // Poll the page for the recorded arrival time.
    let mut latency_ms: Option<u64> = None;
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(20);
    while tokio::time::Instant::now() < deadline && latency_ms.is_none() {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        let resp = execute_command(
            &json!({ "id": "4", "action": "evaluate", "script": "window.__down" }),
            &mut state,
        )
        .await;
        if let Some(down) = get_data(&resp)["result"].as_u64() {
            latency_ms = Some(down.saturating_sub(sent_at_ms));
        }
    }

    let latency = latency_ms.expect("the click should reach the page within the deadline");
    assert!(
        latency < 500,
        "click landed {latency}ms after a 300-event mouse sweep; input dispatch is waiting on CDP replies again"
    );

    let resp = execute_command(
        &json!({ "id": "98", "action": "stream_disable" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
    let _ = std::fs::remove_dir_all(&socket_dir);
}

/// Extract width and height from a JPEG's SOF0 (0xFFC0) or SOF2 (0xFFC2) marker.
fn jpeg_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    for i in 0..data.len().saturating_sub(8) {
        if data[i] == 0xFF && (data[i + 1] == 0xC0 || data[i + 1] == 0xC2) {
            let height = u16::from_be_bytes([data[i + 5], data[i + 6]]) as u32;
            let width = u16::from_be_bytes([data[i + 7], data[i + 8]]) as u32;
            return Some((width, height));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Upload: ref-based selector support (issue #1107)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn e2e_upload_with_ref_selector() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": native_test_fixture_url("upload_probe") }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "3", "action": "snapshot" }), &mut state).await;
    assert_success(&resp);
    let snapshot = get_data(&resp)["snapshot"].as_str().unwrap();

    // Match by label text, not by role which may vary across Chrome versions
    let file_input_ref = snapshot
        .lines()
        .filter_map(|line| {
            if line.contains("Choose file") && line.contains("ref=") {
                let start = line.find("ref=")? + 4;
                let end = line[start..].find(']')? + start;
                Some(line[start..end].to_string())
            } else {
                None
            }
        })
        .next()
        .expect("Snapshot should contain the file input with a ref");

    let tmp = std::env::temp_dir().join(format!("ab-upload-ref-{}.txt", std::process::id()));
    std::fs::write(&tmp, "test").unwrap();

    let resp = execute_command(
        &json!({ "id": "4", "action": "upload", "selector": file_input_ref, "files": [tmp.to_string_lossy()] }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["uploaded"], 1);

    let _ = std::fs::remove_file(&tmp);
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_upload_with_css_selector() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": native_test_fixture_url("upload_probe") }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let tmp = std::env::temp_dir().join(format!("ab-upload-css-{}.txt", std::process::id()));
    std::fs::write(&tmp, "test").unwrap();

    let resp = execute_command(
        &json!({ "id": "3", "action": "upload", "selector": "#fileInput", "files": [tmp.to_string_lossy()] }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["uploaded"], 1);

    let _ = std::fs::remove_file(&tmp);
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Recording: default records the current active page
// ---------------------------------------------------------------------------

/// `recording_start` attaches the recorder to the active page
/// as-is: no new browser context, no new tab, no navigation. Page state set
/// before `record start` must survive, and the viewport must be untouched.
#[tokio::test]
#[ignore]
async fn e2e_recording_default_records_active_page() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let page_url = "data:text/html,<h1>Current</h1>";
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": page_url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "viewport", "width": 800, "height": 600 }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // In-memory page state: a cold navigation or a new page would lose this.
    let resp = execute_command(
        &json!({ "id": "4", "action": "evaluate", "script": "window.__abMarker = 42; window.__abMarker" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], 42);

    let tmp_dir = std::env::temp_dir();
    let rec_path = tmp_dir.join(format!("ab-e2e-rec-current-{}.webm", std::process::id()));
    let resp = execute_command(
        &json!({ "id": "5", "action": "recording_start", "path": rec_path.to_string_lossy() }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    tokio::time::sleep(tokio::time::Duration::from_millis(700)).await;

    // Still exactly one tab: no recording tab was added.
    let resp = execute_command(&json!({ "id": "6", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap();
    assert_eq!(
        tabs.len(),
        1,
        "default record start must not open a new tab, got {tabs:?}"
    );

    // Same page, same JS heap: the marker survived and the URL is unchanged.
    let resp = execute_command(
        &json!({ "id": "7", "action": "evaluate", "script": "window.__abMarker" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["result"],
        42,
        "default record start must not navigate or replace the page"
    );

    let resp = execute_command(
        &json!({ "id": "8", "action": "evaluate", "script": "location.href" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], page_url);

    let resp = execute_command(
        &json!({ "id": "9", "action": "evaluate", "script": "[window.innerWidth, window.innerHeight]" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], json!([800, 600]));

    let resp = execute_command(
        &json!({ "id": "10", "action": "recording_stop" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(
        get_data(&resp)["frames"].as_u64().unwrap_or(0) > 0,
        "recorder attached to the active page should have captured frames"
    );

    let _ = std::fs::remove_file(&rec_path);
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// `recording_start` with a URL navigates the active
/// tab to that URL before recording. No new tab is created.
#[tokio::test]
#[ignore]
async fn e2e_recording_default_with_url_navigates_active_tab() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": "data:text/html,<button id='before'>Before</button>"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Refs from the page being replaced must not survive the navigation.
    let resp = execute_command(
        &json!({ "id": "2b", "action": "snapshot", "selector": "#before" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(state.ref_map.get("e1").is_some());

    let target_url = "data:text/html,<h1>Recorded</h1>";
    let tmp_dir = std::env::temp_dir();
    let rec_path = tmp_dir.join(format!("ab-e2e-rec-url-{}.webm", std::process::id()));
    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "recording_start",
            "path": rec_path.to_string_lossy(),
            "url": target_url
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(
        state.ref_map.entries_sorted().is_empty(),
        "record start <url> must clear refs like navigate does"
    );

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let resp = execute_command(&json!({ "id": "4", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap();
    assert_eq!(
        tabs.len(),
        1,
        "record start <url> must reuse the active tab"
    );

    let resp = execute_command(
        &json!({ "id": "5", "action": "evaluate", "script": "location.href" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], target_url);

    let resp = execute_command(
        &json!({ "id": "6", "action": "recording_stop" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let _ = std::fs::remove_file(&rec_path);
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// Recording: requested frame rate
// ---------------------------------------------------------------------------

/// Verify that a requested frame rate sets the timestamp resolution without
/// filling a static recording with duplicate encoded frames.
#[tokio::test]
#[ignore]
async fn e2e_recording_honors_requested_fps() {
    const FPS: u64 = 60;
    const RECORD_MS: u64 = 1000;

    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "data:text/html,<h1>Frame rate</h1>" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let tmp_dir = std::env::temp_dir();
    let rec_path = tmp_dir.join(format!("ab-e2e-rec-fps-{}.webm", std::process::id()));
    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "recording_start",
            "path": rec_path.to_string_lossy(),
            "fps": FPS,
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["fps"].as_u64(), Some(FPS));

    // The recorder screencasts on its own CDP session attached to the
    // active page. That attachment must not surface as a tab.
    let resp = execute_command(&json!({ "id": "3b", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap().len();
    assert_eq!(tabs, 1, "the recorded page only, nothing else");

    tokio::time::sleep(tokio::time::Duration::from_millis(RECORD_MS)).await;

    let resp = execute_command(
        &json!({ "id": "4", "action": "recording_stop" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let data = get_data(&resp);
    assert_eq!(data["fps"].as_u64(), Some(FPS));

    let frames = data["frames"].as_u64().unwrap();
    assert!(
        (FPS * 8 / 10..FPS * 15 / 10).contains(&frames),
        "static page should repeat frames at {FPS} fps, got {frames}"
    );
    let captured = data["capturedFrames"].as_u64().unwrap();
    assert!(captured >= 1, "static page should produce an initial frame");

    let size = std::fs::metadata(&rec_path).map(|m| m.len()).unwrap_or(0);
    assert!(size > 0, "recording file should not be empty");
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(&rec_path)
        .output()
        .expect("ffprobe should inspect the recording");
    assert!(probe.status.success());
    let duration: f64 = String::from_utf8_lossy(&probe.stdout)
        .trim()
        .parse()
        .expect("ffprobe duration should be numeric");
    assert!(
        (0.8..1.5).contains(&duration),
        "recording should retain wall-clock duration, got {duration}"
    );
    assert!(
        (duration - frames as f64 / FPS as f64).abs() < 0.03,
        "{frames} frames at {FPS} fps should match duration {duration}"
    );

    let _ = std::fs::remove_file(&rec_path);
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Verify that an output path without an extension is rejected before the
/// recording context exists: no new tab, no file, nothing to stop.
#[tokio::test]
#[ignore]
async fn e2e_recording_rejects_extensionless_path_before_context() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let rec_path = std::env::temp_dir().join(format!("ab-e2e-rec-noext-{}", std::process::id()));
    let resp = execute_command(
        &json!({ "id": "2", "action": "recording_start", "path": rec_path.to_string_lossy() }),
        &mut state,
    )
    .await;
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(false));
    let err = resp.get("error").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        err.contains("no extension"),
        "error should explain the path: {}",
        err
    );
    assert!(!rec_path.exists(), "no file should be created");
    assert!(!state.recording_state.active);

    let resp = execute_command(&json!({ "id": "3", "action": "tab_list" }), &mut state).await;
    assert_success(&resp);
    let tabs = get_data(&resp)["tabs"].as_array().unwrap().len();
    assert_eq!(
        tabs, 1,
        "a rejected path must not leave a recording tab behind"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Verify that a missing ffmpeg fails `recording_start` itself before an
/// optional navigation, and leaves the state ready for the next start.
#[tokio::test]
#[ignore]
async fn e2e_recording_fails_fast_without_ffmpeg() {
    let guard = EnvGuard::new(&["PATH"]);
    let original_path = std::env::var("PATH").unwrap_or_default();
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let before_url = "data:text/html,<h1>Before</h1>";
    let after_url = "data:text/html,<h1>After</h1>";
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": before_url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let tmp_dir = std::env::temp_dir();
    let rec_path = tmp_dir.join(format!("ab-e2e-rec-noffmpeg-{}.webm", std::process::id()));

    // A PATH with no ffmpeg on it. Chrome is already running, so nothing else
    // needs resolving.
    let empty_dir = tmp_dir.join(format!("ab-e2e-empty-path-{}", std::process::id()));
    std::fs::create_dir_all(&empty_dir).unwrap();
    guard.set("PATH", &empty_dir.to_string_lossy());

    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "recording_start",
            "path": rec_path.to_string_lossy(),
            "url": after_url
        }),
        &mut state,
    )
    .await;
    assert_eq!(
        resp.get("success").and_then(|v| v.as_bool()),
        Some(false),
        "record start must fail without ffmpeg: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );
    let err = resp.get("error").and_then(|v| v.as_str()).unwrap_or("");
    assert!(err.contains("ffmpeg"), "error should name ffmpeg: {}", err);
    assert!(
        !state.recording_state.active,
        "failed start must not leave the recording active"
    );
    let resp = execute_command(
        &json!({ "id": "4", "action": "evaluate", "script": "location.href" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["result"],
        before_url,
        "ffmpeg preflight must fail before the requested navigation"
    );

    // With ffmpeg back, the same start succeeds: nothing stale was left behind.
    guard.set("PATH", &original_path);
    let resp = execute_command(
        &json!({
            "id": "5",
            "action": "recording_start",
            "path": rec_path.to_string_lossy(),
            "url": after_url
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    let resp = execute_command(
        &json!({ "id": "6", "action": "recording_stop" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let _ = std::fs::remove_file(&rec_path);
    let _ = std::fs::remove_dir(&empty_dir);
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Verify that an out-of-range frame rate is rejected before the recorder
/// builds its context or attaches to the page, leaving no file and no active
/// recording behind.
#[tokio::test]
#[ignore]
async fn e2e_recording_rejects_invalid_fps() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let tmp_dir = std::env::temp_dir();
    let rec_path = tmp_dir.join(format!("ab-e2e-rec-badfps-{}.webm", std::process::id()));
    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "recording_start",
            "path": rec_path.to_string_lossy(),
            "fps": 240,
        }),
        &mut state,
    )
    .await;
    assert_eq!(
        resp.get("success").and_then(|v| v.as_bool()),
        Some(false),
        "240 fps should be rejected: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );
    let err = resp.get("error").and_then(|v| v.as_str()).unwrap_or("");
    assert!(err.contains("fps"), "error should name the field: {}", err);
    assert!(!rec_path.exists(), "no file should be created");

    // Nothing was started, so there is nothing to stop.
    let resp = execute_command(
        &json!({ "id": "3", "action": "recording_stop" }),
        &mut state,
    )
    .await;
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(false));

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Verify the composited pointer and changed-frame contact sheet through the
/// full daemon pipeline.
#[tokio::test]
#[ignore]
async fn e2e_recording_cursor_and_contact_sheet() {
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = r#"data:text/html,<style>body{margin:0;background:%23f3f4f6;font-family:sans-serif}.card{margin:80px;padding:48px;background:white;border-radius:24px}button{padding:18px 28px;background:%232563eb;color:white;border:0;border-radius:12px}</style><div class=card><h1>Contact sheet demo</h1><p>Review important visual changes at a glance.</p><button>Continue</button></div>"#;
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": html }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let rec_path = std::env::temp_dir().join(format!(
        "ab-e2e-rec-contact-sheet-{}.webm",
        std::process::id()
    ));
    let sheet_path = rec_path.with_file_name(format!(
        "{}.contact-sheet.png",
        rec_path.file_stem().unwrap().to_string_lossy()
    ));
    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "recording_start",
            "path": rec_path.to_string_lossy(),
            "cursor": true,
            "contactSheet": true,
            "contactSheetThreshold": 0.01
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["cursor"], true);
    assert_eq!(get_data(&resp)["contactSheet"], true);

    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
    let resp = execute_command(
        &json!({ "id": "4", "action": "evaluate", "script": "Boolean(document.getElementById('__agent_browser_recording_cursor__'))" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], false);

    let resp = execute_command(
        &json!({ "id": "5", "action": "mousemove", "x": 360, "y": 260 }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let cursor = state
        .recording_state
        .shared_cursor
        .lock()
        .unwrap()
        .at(super::recording::cursor_timestamp());
    assert!(cursor.visible);
    assert_eq!((cursor.x, cursor.y), (360.0, 260.0));
    let resp = execute_command(
        &json!({ "id": "6", "action": "evaluate", "script": "document.querySelector('.card').style.background='#dbeafe'; document.querySelector('h1').textContent='Ready to continue'; true" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;
    let resp = execute_command(
        &json!({ "id": "7", "action": "evaluate", "script": "document.querySelector('.card').style.background='#dcfce7'; document.querySelector('p').textContent='The important region is highlighted.'; true" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;

    let resp = execute_command(
        &json!({ "id": "8", "action": "recording_stop" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let data = get_data(&resp);
    assert_eq!(
        data["contactSheetPath"],
        sheet_path.to_string_lossy().as_ref()
    );
    assert!(data["contactSheetFrames"].as_u64().unwrap_or(0) >= 2);
    assert!(std::fs::metadata(&rec_path).unwrap().len() > 0);
    let sheet = image::open(&sheet_path).expect("contact sheet should be a valid PNG");
    assert!(sheet.width() >= 320);
    assert!(sheet.height() >= 150);
    if let Some(example_path) = std::env::var_os("AGENT_BROWSER_CONTACT_SHEET_EXAMPLE_PATH") {
        let example_path = std::path::PathBuf::from(example_path);
        if let Some(parent) = example_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::copy(&sheet_path, example_path).unwrap();
    }

    let resp = execute_command(
        &json!({ "id": "9", "action": "evaluate", "script": "Boolean(document.getElementById('__agent_browser_recording_cursor__'))" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], false);

    let _ = std::fs::remove_file(&rec_path);
    let _ = std::fs::remove_file(&sheet_path);
    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// The cursor must not intercept clicks, leak into snapshots, or survive stop.
#[tokio::test]
#[ignore]
async fn e2e_recording_cursor_overlay_lifecycle() {
    let mut state = DaemonState::new();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cursor-lifecycle.mp4");
    let url =
        "data:text/html,<button id=keep onclick='window.clicks=(window.clicks||0)+1'>Keep</button>";
    for command in [
        json!({"action":"launch","headless":true}),
        json!({"action":"navigate","url":url}),
        json!({"action":"recording_start","path":path,"cursor":true}),
        json!({"action":"click","selector":"#keep"}),
    ] {
        assert_success(&execute_command(&command, &mut state).await);
    }
    let inspected = execute_command(&json!({"action":"evaluate","script":"(() => { const host = document.querySelector('[data-agent-browser-recording-cursor]'); return !!host && host.inert && host.getAttribute('aria-hidden') === 'true' && host.shadowRoot === null && typeof globalThis.__agentBrowserRecordingCursorCleanup === 'undefined' && window.clicks === 1; })()"}), &mut state).await;
    assert_success(&inspected);
    assert_eq!(get_data(&inspected)["result"], true);
    let snapshot = execute_command(&json!({"action":"snapshot"}), &mut state).await;
    assert_success(&snapshot);
    assert!(!get_data(&snapshot)
        .to_string()
        .contains("agent-browser-recording-cursor"));
    assert_success(
        &execute_command(
            &json!({"action":"navigate","url":"data:text/html,<button>After navigation</button>"}),
            &mut state,
        )
        .await,
    );
    let after_navigation = execute_command(&json!({"action":"evaluate","script":"document.querySelectorAll('[data-agent-browser-recording-cursor]').length"}), &mut state).await;
    let stopped = execute_command(&json!({"action":"recording_stop"}), &mut state).await;
    let after_stop = execute_command(&json!({"action":"evaluate","script":"document.querySelectorAll('[data-agent-browser-recording-cursor]').length"}), &mut state).await;
    assert_success(&execute_command(&json!({"action":"navigate","url":url}), &mut state).await);
    let after_next_navigation = execute_command(&json!({"action":"evaluate","script":"document.querySelectorAll('[data-agent-browser-recording-cursor]').length"}), &mut state).await;
    assert_success(&execute_command(&json!({"action":"close"}), &mut state).await);
    assert_success(&stopped);
    assert_eq!(get_data(&after_navigation)["result"], 1);
    assert_eq!(get_data(&after_stop)["result"], 0);
    assert_eq!(get_data(&after_next_navigation)["result"], 0);
}

/// Check every encoded frame while a real page control follows drag input.
#[tokio::test]
#[ignore]
async fn e2e_recording_cursor_stays_aligned_during_drag() {
    let mut state = DaemonState::new();
    for command in [
        json!({ "action": "launch", "headless": true }),
        json!({ "action": "viewport", "width": 640, "height": 480 }),
        json!({
            "action": "navigate",
            "url": "data:text/html,<style>body{margin:0;background:%23303030}div{position:fixed;left:80px;top:0;width:2px;height:100vh;background:%2300ff00}</style><div></div><script>document.addEventListener('pointermove',e=>{if(e.buttons)document.querySelector('div').style.left=e.clientX+'px'})</script>"
        }),
        json!({ "action": "mousemove", "x": 80, "y": 240 }),
    ] {
        assert_success(&execute_command(&command, &mut state).await);
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("drag-cursor.mp4");
    assert_success(
        &execute_command(
            &json!({ "action": "recording_start", "path": path, "cursor": true, "fps": 60 }),
            &mut state,
        )
        .await,
    );
    assert_success(&execute_command(&json!({ "action": "mousedown" }), &mut state).await);
    for x in [560, 80, 560] {
        assert_success(
            &execute_command(
                &json!({ "action": "mousemove", "x": x, "y": 240, "duration": 1000, "inputMode": "human", "seed": 42 }),
                &mut state,
            )
            .await,
        );
    }
    assert_success(&execute_command(&json!({ "action": "mouseup" }), &mut state).await);
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_success(&execute_command(&json!({ "action": "recording_stop" }), &mut state).await);
    assert_success(&execute_command(&json!({ "action": "close" }), &mut state).await);

    let output = tokio::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args([
            "-vf",
            "crop=640:80:0:200",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .output()
        .await
        .unwrap();
    assert!(output.status.success());
    let mut measured = 0;
    let mut worst = 0;
    let mut positions = std::collections::HashSet::new();
    for bytes in output.stdout.chunks_exact(640 * 80 * 3) {
        let frame = image::RgbImage::from_raw(640, 80, bytes.to_vec()).unwrap();
        let line = (0..640).find(|&x| {
            let [r, g, b] = frame.get_pixel(x, 0).0;
            g > 150 && g > r.saturating_add(60) && g > b.saturating_add(60)
        });
        let cursor = frame
            .enumerate_pixels()
            .filter_map(|(x, _, pixel)| {
                let [r, g, b] = pixel.0;
                (r > 180 && g > 180 && b > 180).then_some(x)
            })
            .min();
        if let (Some(line), Some(cursor)) = (line, cursor) {
            measured += 1;
            positions.insert(line);
            worst = worst.max(line.abs_diff(cursor));
        }
    }
    assert!(measured >= 100, "only {measured} drag frames measured");
    assert!(
        positions.len() >= 60,
        "drag must exercise moving page frames"
    );
    // The cursor's white fill starts 1-2 pixels inside its black outline.
    assert!(
        worst <= 3,
        "recorded cursor separated from the dragged control by {worst} pixels"
    );
}

/// A completed drag must not pin the recorded cursor to a stale page frame.
#[tokio::test]
#[ignore]
async fn e2e_recording_cursor_moves_after_release_without_repaint() {
    let mut state = DaemonState::new();
    assert_success(
        &execute_command(&json!({ "action": "launch", "headless": true }), &mut state).await,
    );
    assert_success(
        &execute_command(
            &json!({ "action": "viewport", "width": 640, "height": 480 }),
            &mut state,
        )
        .await,
    );
    assert_success(
        &execute_command(
            &json!({
                "action": "navigate",
                "url": "data:text/html,<style>body{margin:0;background:%23202020}</style>"
            }),
            &mut state,
        )
        .await,
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("released-cursor.mp4");
    assert_success(
        &execute_command(
            &json!({ "action": "recording_start", "path": path, "cursor": true, "fps": 60 }),
            &mut state,
        )
        .await,
    );
    for command in [
        json!({ "action": "mousemove", "x": 80, "y": 80 }),
        json!({ "action": "mousedown" }),
        json!({ "action": "mousemove", "x": 120, "y": 100, "duration": 150, "inputMode": "human" }),
    ] {
        assert_success(&execute_command(&command, &mut state).await);
    }
    let captured = state
        .recording_state
        .shared_captured_count
        .as_ref()
        .unwrap()
        .clone();
    let before = captured.load(Ordering::Relaxed);
    // Paint while pressed, then leave the page completely static after release.
    assert_success(
        &execute_command(
            &json!({ "action": "evaluate", "script": "document.body.style.background = '#303030'" }),
            &mut state,
        )
        .await,
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while captured.load(Ordering::Relaxed) == before {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the pressed page frame should reach the recorder");
    for command in [
        json!({ "action": "mouseup" }),
        json!({ "action": "mousemove", "x": 240, "y": 180, "duration": 250, "inputMode": "human" }),
    ] {
        assert_success(&execute_command(&command, &mut state).await);
    }
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    assert_success(&execute_command(&json!({ "action": "recording_stop" }), &mut state).await);
    assert_success(&execute_command(&json!({ "action": "close" }), &mut state).await);

    let output = tokio::process::Command::new("ffmpeg")
        .args(["-v", "error", "-sseof", "-0.1", "-i"])
        .arg(&path)
        .args([
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "pipe:1",
        ])
        .output()
        .await
        .unwrap();
    assert!(output.status.success());
    let frame = image::load_from_memory(&output.stdout).unwrap().to_rgb8();
    assert_eq!(frame.dimensions(), (640, 480));
    assert!(
        frame.get_pixel(244, 184).0.iter().all(|value| *value > 180),
        "the released cursor should reach the new position in the encoded video; got {:?}",
        frame.get_pixel(244, 184).0
    );
    assert!(
        frame.get_pixel(124, 104).0.iter().all(|value| *value < 80),
        "the old drag position should contain only the page background"
    );
}

#[tokio::test]
#[ignore]
async fn e2e_initial_recording_frame_uses_css_viewport_dimensions() {
    let mut state = DaemonState::new();
    assert_success(
        &execute_command(&json!({ "action": "launch", "headless": true }), &mut state).await,
    );
    assert_success(
        &execute_command(
            &json!({
                "action": "viewport",
                "width": 400,
                "height": 300,
                "deviceScaleFactor": 2.0
            }),
            &mut state,
        )
        .await,
    );

    let browser = state.browser.as_ref().unwrap();
    let initial = super::recording::capture_initial_image(
        &browser.client,
        browser.active_session_id().unwrap(),
    )
    .await
    .unwrap();
    let (pixel_width, pixel_height) = image::ImageReader::with_format(
        std::io::Cursor::new(&initial.image_data),
        image::ImageFormat::Png,
    )
    .into_dimensions()
    .unwrap();

    assert_eq!((pixel_width, pixel_height), (800, 600));
    assert_eq!(
        (initial.device_width, initial.device_height),
        (400.0, 300.0)
    );

    assert_success(&execute_command(&json!({ "action": "close" }), &mut state).await);
}

// ---------------------------------------------------------------------------
// tab new: session setup inheritance
// ---------------------------------------------------------------------------

/// `tab new <url>` must replay the session's setup onto the new tab before
/// its first document: an init script registered on the primary page has to
/// run on the initial load, not only after a later navigation.
#[tokio::test]
#[ignore]
async fn e2e_tab_new_inherits_init_script_on_first_load() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "addinitscript", "script": "window.__abTab = 'seeded';" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "3", "action": "tab_new",
            "url": "data:text/html,<script>document.title = String(window.__abTab)</script>",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "4", "action": "evaluate", "script": "document.title" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["result"],
        "seeded",
        "init script should run on the new tab's first document"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Replayed init scripts receive target-specific CDP identifiers. Removing a
/// script from the new tab must translate the original user-facing identifier
/// for every tab where Chrome registered it.
#[tokio::test]
#[ignore]
async fn e2e_tab_new_removes_replayed_init_script_by_original_identifier() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let first = execute_command(
        &json!({
            "id": "2", "action": "addinitscript",
            "script": "window.__abFirstInit = true;",
        }),
        &mut state,
    )
    .await;
    assert_success(&first);
    let first_id = get_data(&first)["identifier"]
        .as_str()
        .expect("first init script should return an identifier")
        .to_string();

    let second = execute_command(
        &json!({
            "id": "3", "action": "addinitscript",
            "script": "window.__abSecondInit = true;",
        }),
        &mut state,
    )
    .await;
    assert_success(&second);
    let second_id = get_data(&second)["identifier"]
        .as_str()
        .expect("second init script should return an identifier")
        .to_string();

    let resp = execute_command(
        &json!({ "id": "4", "action": "removeinitscript", "identifier": first_id }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "5", "action": "tab_new",
            "url": "data:text/html,<title>new tab</title>",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "6", "action": "removeinitscript", "identifier": second_id }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "7", "action": "navigate",
            "url": "data:text/html,<title>after removal</title>",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "8", "action": "evaluate",
            "script": "window.__abSecondInit === true",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], false);

    let resp = execute_command(
        &json!({ "id": "9", "action": "tab_switch", "tabId": "t1" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "10", "action": "navigate",
            "url": "data:text/html,<title>original after removal</title>",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "11", "action": "evaluate",
            "script": "window.__abSecondInit === true",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["result"],
        false,
        "removing a replayed script should also remove its original registration"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// CDP allocates init-script identifiers independently in each target. Two
/// pre-existing tabs can therefore both return `1` for different scripts, but
/// the daemon must expose distinct handles and remove only the requested one.
#[tokio::test]
#[ignore]
async fn e2e_tab_init_script_handles_are_unique_across_preexisting_tabs() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "2", "action": "tab_new" }), &mut state).await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "tab_switch", "tabId": "t1" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let first = execute_command(
        &json!({
            "id": "4", "action": "addinitscript",
            "script": "window.__abFirstExistingTab = true;",
        }),
        &mut state,
    )
    .await;
    assert_success(&first);
    let first_id = get_data(&first)["identifier"]
        .as_str()
        .expect("first init script should return an identifier")
        .to_string();

    let resp = execute_command(
        &json!({ "id": "5", "action": "tab_switch", "tabId": "t2" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let second = execute_command(
        &json!({
            "id": "6", "action": "addinitscript",
            "script": "window.__abSecondExistingTab = true;",
        }),
        &mut state,
    )
    .await;
    assert_success(&second);
    let second_id = get_data(&second)["identifier"]
        .as_str()
        .expect("second init script should return an identifier")
        .to_string();

    assert_ne!(first_id, second_id, "user-facing handles must be unique");

    let resp = execute_command(
        &json!({ "id": "7", "action": "removeinitscript", "identifier": second_id }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "8", "action": "tab_new",
            "url": "data:text/html,<title>future tab</title>",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "9", "action": "evaluate",
            "script": "[window.__abFirstExistingTab === true, window.__abSecondExistingTab === true]",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], json!([true, false]));

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// `click --new-tab` creates a tab through a separate handler from `tab new`,
/// but it must apply the same session setup before the first request.
#[tokio::test]
#[ignore]
async fn e2e_click_new_tab_inherits_user_agent_and_headers() {
    let (base_url, _server) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({
            "id": "1", "action": "launch", "headless": true,
            "userAgent": "ab-click-new-tab-test/1.0",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "headers", "headers": { "X-Global": "global" } }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "3", "action": "navigate",
            "url": format!("data:text/html,<a id='next' href='{}/click'>next</a>", base_url),
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "4", "action": "click", "selector": "#next", "newTab": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "5", "action": "evaluate",
            "script": "JSON.parse(document.body.innerText)",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let headers = &get_data(&resp)["result"]["headers"];
    assert_eq!(headers["X-Global"], "global");
    assert_eq!(headers["User-Agent"], "ab-click-new-tab-test/1.0");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// Clearing headers and offline mode restores the default setup, so future
/// tabs can use the non-blocking target creation path.
#[tokio::test]
#[ignore]
async fn e2e_cleared_session_setup_is_not_pending() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "offline", "offline": false }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "headers", "headers": {} }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    assert!(state.session_setup.offline.is_none());
    assert!(state.session_setup.extra_headers.is_none());

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// The launch `--user-agent` (Emulation.setUserAgentOverride) and global
/// `set headers` (Network.setExtraHTTPHeaders) are per CDP session. A new tab
/// opened with a URL must send both on its very first document request.
#[tokio::test]
#[ignore]
async fn e2e_tab_new_inherits_user_agent_and_headers() {
    let (base_url, _server) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({
            "id": "1", "action": "launch", "headless": true,
            "userAgent": "ab-tab-new-test/1.0",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "headers", "headers": { "X-Global": "global" } }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "tab_new", "url": format!("{}/tab", base_url) }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "4", "action": "evaluate",
            "script": "JSON.parse(document.body.innerText)",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let headers = &get_data(&resp)["result"]["headers"];
    assert_eq!(
        headers["X-Global"], "global",
        "global `headers` should apply to the new tab's first document request, got {headers}"
    );
    assert_eq!(
        headers["User-Agent"], "ab-tab-new-test/1.0",
        "new tab's first document request should carry the launch user agent, got {headers}"
    );

    let resp = execute_command(
        &json!({ "id": "5", "action": "evaluate", "script": "navigator.userAgent" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["result"], "ab-tab-new-test/1.0");

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

/// `set credentials` uses target-scoped extra headers, so a new tab must send
/// the resulting Authorization header on its first document request.
#[tokio::test]
#[ignore]
async fn e2e_tab_new_inherits_http_credentials_on_first_load() {
    let (base_url, _server) = start_echo_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2", "action": "credentials",
            "username": "tab-user", "password": "tab-password",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "tab_new", "url": format!("{}/credentials", base_url) }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "4", "action": "evaluate",
            "script": "JSON.parse(document.body.innerText)",
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let expected = format!("Basic {}", STANDARD.encode("tab-user:tab-password"));
    assert_eq!(
        get_data(&resp)["result"]["headers"]["Authorization"],
        expected,
        "HTTP credentials should apply to the new tab's first document request"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);
}

// ---------------------------------------------------------------------------
// --state / storageState flag: cookies should be loaded at launch time
// ---------------------------------------------------------------------------

/// Verify that launching with `storageState` in the launch command restores
/// cookies that were previously saved with `state_save`.
///
/// This is the e2e equivalent of `agent-browser --state ./auth.json open <url>`.
/// The launch command accepts a `storageState` field that should load the
/// state file (cookies + localStorage) before the first navigation.
#[tokio::test]
#[ignore]
async fn e2e_state_flag_restores_cookies() {
    let state_path = std::env::temp_dir()
        .join(format!(
            "agent-browser-e2e-state-flag-{}.json",
            uuid::Uuid::new_v4()
        ))
        .to_string_lossy()
        .to_string();

    // Session 1: launch, set a cookie, save state, close
    {
        let mut state = DaemonState::new();

        let resp = execute_command(
            &json!({ "id": "1", "action": "launch", "headless": true }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "cookies_set",
                "name": "state_flag_test",
                "value": "from_state_file",
                "domain": ".example.com",
                "path": "/",
                "expires": 2000000000
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "4", "action": "state_save", "path": &state_path }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(&json!({ "id": "5", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    // Session 2: launch with storageState pointing to saved file, verify
    // cookies are present before any explicit state_load call.
    {
        let mut state = DaemonState::new();

        let resp = execute_command(
            &json!({
                "id": "10",
                "action": "launch",
                "headless": true,
                "storageState": &state_path
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "11", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp =
            execute_command(&json!({ "id": "12", "action": "cookies_get" }), &mut state).await;
        assert_success(&resp);
        let cookies = get_data(&resp)["cookies"].as_array().unwrap();
        let found = cookies
            .iter()
            .any(|c| c["name"] == "state_flag_test" && c["value"] == "from_state_file");
        assert!(
            found,
            "Cookie from state file should be present after launch with storageState. \
             Cookies found: {:?}",
            cookies
                .iter()
                .map(|c| c["name"].as_str().unwrap_or("?"))
                .collect::<Vec<_>>()
        );

        let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    let _ = std::fs::remove_file(&state_path);
}

/// Verify that explicit `launch` surfaces storageState load failures instead
/// of reporting success with an empty browser state.
#[tokio::test]
#[ignore]
async fn e2e_state_flag_missing_file_fails_launch() {
    let guard = EnvGuard::new(&["CI"]);
    guard.set("CI", "1");

    let missing_path = std::env::temp_dir()
        .join(format!(
            "agent-browser-e2e-missing-state-{}.json",
            uuid::Uuid::new_v4()
        ))
        .to_string_lossy()
        .to_string();

    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({
            "id": "10",
            "action": "launch",
            "headless": true,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            "storageState": &missing_path
        }),
        &mut state,
    )
    .await;

    assert_eq!(resp["success"], false);
    let error = resp["error"].as_str().unwrap_or_default();
    assert!(
        error.contains("Failed to read state from") || error.contains("storage state"),
        "Unexpected error for missing storageState file: {}",
        error
    );
    assert!(
        state.browser.is_none(),
        "failed storageState launch should roll back the browser"
    );
}

/// Repeated launch calls with `storageState` should relaunch a clean browser so
/// stale cookies do not survive from the previous state file.
#[tokio::test]
#[ignore]
async fn e2e_storage_state_launch_restarts_clean_browser() {
    let state_one = std::env::temp_dir()
        .join(format!(
            "agent-browser-e2e-storage-reuse-1-{}.json",
            uuid::Uuid::new_v4()
        ))
        .to_string_lossy()
        .to_string();
    let state_two = std::env::temp_dir()
        .join(format!(
            "agent-browser-e2e-storage-reuse-2-{}.json",
            uuid::Uuid::new_v4()
        ))
        .to_string_lossy()
        .to_string();

    create_storage_state_with_cookie(&state_one, "storage_reload_first", "first").await;
    create_storage_state_with_cookie(&state_two, "storage_reload_second", "second").await;

    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({
            "id": "10",
            "action": "launch",
            "headless": true,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            "storageState": &state_one
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(
        get_data(&resp).get("reused").is_none(),
        "first launch must create the browser"
    );

    let resp = execute_command(
        &json!({ "id": "11", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "12", "action": "cookies_get" }), &mut state).await;
    assert_success(&resp);
    let cookies = get_data(&resp)["cookies"].as_array().unwrap();
    assert!(
        cookies
            .iter()
            .any(|c| c["name"] == "storage_reload_first" && c["value"] == "first"),
        "first storageState should be applied on the initial launch"
    );

    let resp = execute_command(
        &json!({
            "id": "13",
            "action": "launch",
            "headless": true,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            "storageState": &state_two
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp).get("reused"),
        None,
        "storageState launch should start from a clean browser"
    );

    let resp = execute_command(
        &json!({ "id": "14", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "15", "action": "cookies_get" }), &mut state).await;
    assert_success(&resp);
    let cookies = get_data(&resp)["cookies"].as_array().unwrap();
    assert!(
        cookies
            .iter()
            .any(|c| c["name"] == "storage_reload_second" && c["value"] == "second"),
        "second storageState should be applied after relaunch"
    );
    assert!(
        !cookies
            .iter()
            .any(|c| c["name"] == "storage_reload_first" && c["value"] == "first"),
        "stale cookies from the first storageState should not survive relaunch"
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);

    let _ = std::fs::remove_file(&state_one);
    let _ = std::fs::remove_file(&state_two);
}

/// Verify that AGENT_BROWSER_STATE env var restores cookies at auto-launch
/// time (when the browser is lazily launched by a command like `navigate`
/// rather than an explicit `launch` command).
#[tokio::test]
#[ignore]
async fn e2e_state_env_restores_cookies_on_auto_launch() {
    let state_path = std::env::temp_dir()
        .join(format!(
            "agent-browser-e2e-state-env-{}.json",
            uuid::Uuid::new_v4()
        ))
        .to_string_lossy()
        .to_string();

    // Session 1: launch, set a cookie, save state, close
    {
        let mut state = DaemonState::new();

        let resp = execute_command(
            &json!({ "id": "1", "action": "launch", "headless": true }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "cookies_set",
                "name": "env_state_test",
                "value": "from_env_state",
                "domain": ".example.com",
                "path": "/",
                "expires": 2000000000
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "4", "action": "state_save", "path": &state_path }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(&json!({ "id": "5", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    // Session 2: set AGENT_BROWSER_STATE env var and let auto_launch pick it
    // up. No explicit `launch` command — just navigate, which triggers
    // auto_launch internally.
    {
        let env = EnvGuard::new(&["AGENT_BROWSER_STATE"]);
        env.set("AGENT_BROWSER_STATE", &state_path);

        let mut state = DaemonState::new();

        // Navigate without explicit launch — triggers auto_launch
        let resp = execute_command(
            &json!({ "id": "10", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp =
            execute_command(&json!({ "id": "11", "action": "cookies_get" }), &mut state).await;
        assert_success(&resp);
        let cookies = get_data(&resp)["cookies"].as_array().unwrap();
        let found = cookies
            .iter()
            .any(|c| c["name"] == "env_state_test" && c["value"] == "from_env_state");
        assert!(
            found,
            "Cookie should be restored via AGENT_BROWSER_STATE env on auto-launch. \
             Cookies found: {:?}",
            cookies
                .iter()
                .map(|c| c["name"].as_str().unwrap_or("?"))
                .collect::<Vec<_>>()
        );

        let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    let _ = std::fs::remove_file(&state_path);
}

/// Verify that --session-name auto-restores cookies saved from a prior
/// session with the same name.
#[tokio::test]
#[ignore]
async fn e2e_session_name_auto_restores_cookies() {
    let session_name = format!(
        "e2e-session-name-{}",
        &uuid::Uuid::new_v4().to_string()[..8]
    );

    let env = EnvGuard::new(&["AGENT_BROWSER_SESSION_NAME"]);
    env.set("AGENT_BROWSER_SESSION_NAME", &session_name);

    // Session 1: launch, set a cookie, close (which auto-saves state)
    {
        let mut state = DaemonState::new();

        let resp = execute_command(
            &json!({ "id": "1", "action": "launch", "headless": true }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "cookies_set",
                "name": "session_name_test",
                "value": "auto_restored",
                "domain": ".example.com",
                "path": "/",
                "expires": 2000000000
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        // close triggers auto-save when session_name is set
        let resp = execute_command(&json!({ "id": "5", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    // Session 2: fresh DaemonState with same session_name. Navigate without
    // explicit launch, which triggers auto_launch and restore.
    {
        let mut state = DaemonState::new();

        // Navigate without explicit launch — triggers auto_launch → try_auto_restore_state
        let resp = execute_command(
            &json!({ "id": "10", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp =
            execute_command(&json!({ "id": "12", "action": "cookies_get" }), &mut state).await;
        assert_success(&resp);
        let cookies = get_data(&resp)["cookies"].as_array().unwrap();
        let found = cookies
            .iter()
            .any(|c| c["name"] == "session_name_test" && c["value"] == "auto_restored");
        assert!(
            found,
            "Cookie should be auto-restored via --session-name. Cookies found: {:?}",
            cookies
                .iter()
                .map(|c| c["name"].as_str().unwrap_or("?"))
                .collect::<Vec<_>>()
        );

        let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    // Clean up auto-saved state files
    let sessions_dir = dirs::home_dir()
        .unwrap()
        .join(".agent-browser")
        .join("sessions");
    if let Ok(entries) = std::fs::read_dir(&sessions_dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname.starts_with(&format!("{}-", session_name)) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

#[tokio::test]
#[ignore]
async fn e2e_restore_loads_during_explicit_launch_before_navigation() {
    let restore_key = format!(
        "e2e-explicit-restore-{}",
        &uuid::Uuid::new_v4().to_string()[..8]
    );
    let env = EnvGuard::new(&[
        "AGENT_BROWSER_SESSION_NAME",
        "AGENT_BROWSER_RESTORE_SAVE",
        "AGENT_BROWSER_STATE",
        "AGENT_BROWSER_ENCRYPTION_KEY",
    ]);
    env.remove("AGENT_BROWSER_SESSION_NAME");
    env.remove("AGENT_BROWSER_RESTORE_SAVE");
    env.remove("AGENT_BROWSER_STATE");
    env.remove("AGENT_BROWSER_ENCRYPTION_KEY");

    {
        let mut state = DaemonState::new();
        let resp = execute_command(
            &json!({
                "id": "1",
                "action": "launch",
                "headless": true,
                "restoreKey": restore_key
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "cookies_set",
                "name": "explicit_restore_test",
                "value": "loaded_before_navigation",
                "domain": ".example.com",
                "path": "/",
                "expires": 2000000000
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(&json!({ "id": "4", "action": "close" }), &mut state).await;
        assert_success(&resp);
        assert_eq!(get_data(&resp)["saveStatus"], "saved");
    }

    let path = super::state::find_auto_state_file(&restore_key)
        .expect("first close should create restore state");

    {
        let mut state = DaemonState::new();
        let resp = execute_command(
            &json!({
                "id": "10",
                "action": "launch",
                "headless": true,
                "restoreKey": restore_key
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert_eq!(get_data(&resp)["lifecycle"]["restoreStatus"], "loaded");

        let resp = execute_command(
            &json!({ "id": "11", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp =
            execute_command(&json!({ "id": "12", "action": "cookies_get" }), &mut state).await;
        assert_success(&resp);
        let cookies = get_data(&resp)["cookies"].as_array().unwrap();
        let found = cookies.iter().any(|c| {
            c["name"] == "explicit_restore_test" && c["value"] == "loaded_before_navigation"
        });
        assert!(
            found,
            "Cookie should be restored by explicit launch before first navigation. Cookies found: {:?}",
            cookies
                .iter()
                .map(|c| c["name"].as_str().unwrap_or("?"))
                .collect::<Vec<_>>()
        );

        let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.previous", path));
}

/// Verify that periodic autosave persists session state while the browser is
/// open, so state survives the browser dying WITHOUT the daemon's
/// save-on-close path running (e.g. the user closes the window by hand).
#[tokio::test]
#[ignore]
async fn e2e_periodic_autosave_survives_abrupt_browser_exit() {
    let restore_key = format!(
        "e2e-autosave-abrupt-{}",
        &uuid::Uuid::new_v4().to_string()[..8]
    );
    let env = EnvGuard::new(&[
        "AGENT_BROWSER_SESSION_NAME",
        "AGENT_BROWSER_RESTORE_SAVE",
        "AGENT_BROWSER_STATE",
        "AGENT_BROWSER_ENCRYPTION_KEY",
    ]);
    env.remove("AGENT_BROWSER_SESSION_NAME");
    env.remove("AGENT_BROWSER_RESTORE_SAVE");
    env.remove("AGENT_BROWSER_STATE");
    env.remove("AGENT_BROWSER_ENCRYPTION_KEY");

    {
        let mut state = DaemonState::new();
        let resp = execute_command(
            &json!({
                "id": "1",
                "action": "launch",
                "headless": true,
                "restoreKey": restore_key
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "cookies_set",
                "name": "autosave_test",
                "value": "saved_by_tick",
                "domain": ".example.com",
                "path": "/",
                "expires": 2000000000
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        // The command just finished, so the quiet period must block the tick.
        maybe_autosave_restore_state(&mut state, 30_000).await;
        assert_eq!(state.restore_save_status, "not_attempted");
        assert!(
            super::state::find_auto_state_file(&restore_key).is_none(),
            "autosave must not fire inside the post-command quiet period"
        );

        // Simulate the quiet period having elapsed, then run the tick again.
        state.last_command_finished =
            std::time::Instant::now().checked_sub(std::time::Duration::from_secs(10));
        maybe_autosave_restore_state(&mut state, 30_000).await;
        assert_eq!(state.restore_save_status, "saved");
        assert!(
            state.last_autosave_attempt.is_some(),
            "successful save should reset the periodic interval"
        );
        assert!(
            super::state::find_auto_state_file(&restore_key).is_some(),
            "periodic autosave should write the session state file"
        );

        // An idle session stays eligible: once the interval elapses again the
        // tick re-saves, capturing page-driven mutations like token refreshes.
        state.last_autosave_attempt =
            std::time::Instant::now().checked_sub(std::time::Duration::from_secs(31));
        state.restore_save_status = "not_attempted".to_string();
        maybe_autosave_restore_state(&mut state, 30_000).await;
        assert_eq!(
            state.restore_save_status, "saved",
            "idle session should be re-saved on the next interval without new commands"
        );

        // Kill the browser out from under the daemon, the way a user closing
        // the window does: the process exits and CDP dies, so no further save
        // is possible. Then mimic the daemon drain tick, which only closes.
        let mgr = state.browser.as_mut().expect("browser should be running");
        let _ = mgr
            .client
            .send_command_no_params("Browser.close", None)
            .await;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !mgr.has_process_exited() && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(
            state
                .browser
                .as_mut()
                .expect("browser manager should still be present")
                .has_process_exited(),
            "browser process should have exited after Browser.close"
        );
        let _ = close_current_browser(&mut state).await;
    }

    let path = super::state::find_auto_state_file(&restore_key)
        .expect("autosaved state should survive the abrupt browser exit");

    {
        let mut state = DaemonState::new();
        let resp = execute_command(
            &json!({
                "id": "10",
                "action": "launch",
                "headless": true,
                "restoreKey": restore_key
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert_eq!(get_data(&resp)["lifecycle"]["restoreStatus"], "loaded");

        let resp = execute_command(
            &json!({ "id": "11", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp =
            execute_command(&json!({ "id": "12", "action": "cookies_get" }), &mut state).await;
        assert_success(&resp);
        let cookies = get_data(&resp)["cookies"].as_array().unwrap();
        let found = cookies
            .iter()
            .any(|c| c["name"] == "autosave_test" && c["value"] == "saved_by_tick");
        assert!(
            found,
            "Cookie saved only by periodic autosave should be restored after the browser was killed without a graceful close. Cookies found: {:?}",
            cookies
                .iter()
                .map(|c| c["name"].as_str().unwrap_or("?"))
                .collect::<Vec<_>>()
        );

        let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.previous", path));
}

#[tokio::test]
#[ignore]
async fn e2e_restore_preserves_cookie_login_after_close_and_reopen() {
    let restore_key = format!(
        "e2e-next-cookie-login-{}",
        &uuid::Uuid::new_v4().to_string()[..8]
    );
    let env = EnvGuard::new(&[
        "AGENT_BROWSER_SESSION_NAME",
        "AGENT_BROWSER_RESTORE_SAVE",
        "AGENT_BROWSER_STATE",
        "AGENT_BROWSER_ENCRYPTION_KEY",
    ]);
    env.remove("AGENT_BROWSER_SESSION_NAME");
    env.remove("AGENT_BROWSER_RESTORE_SAVE");
    env.remove("AGENT_BROWSER_STATE");
    env.remove("AGENT_BROWSER_ENCRYPTION_KEY");

    let (base_url, _server) = start_cookie_login_server().await;

    {
        let mut state = DaemonState::new();

        let resp = execute_command(
            &json!({
                "id": "1",
                "action": "navigate",
                "url": base_url.clone(),
                "restoreKey": restore_key
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert_eq!(get_data(&resp)["lifecycle"]["restoreStatus"], "missing");

        let resp = execute_command(
            &json!({ "id": "2", "action": "evaluate", "script": "document.body.innerText" }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert!(
            get_data(&resp)["result"]
                .as_str()
                .unwrap_or_default()
                .contains("Please sign in"),
            "first homepage load should be logged out: {}",
            resp
        );

        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "navigate",
                "url": format!("{}/login", base_url),
                "restoreKey": restore_key
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({
                "id": "4",
                "action": "navigate",
                "url": base_url.clone(),
                "restoreKey": restore_key
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "5", "action": "evaluate", "script": "document.body.innerText" }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert!(
            get_data(&resp)["result"]
                .as_str()
                .unwrap_or_default()
                .contains("Welcome back"),
            "login route should set cookie for current browser: {}",
            resp
        );

        let resp = execute_command(&json!({ "id": "6", "action": "close" }), &mut state).await;
        assert_success(&resp);
        assert_eq!(get_data(&resp)["saveStatus"], "saved");
    }

    let path = super::state::find_auto_state_file(&restore_key)
        .expect("close should save cookie-backed restore state");

    {
        let mut state = DaemonState::new();
        let resp = execute_command(
            &json!({
                "id": "10",
                "action": "navigate",
                "url": base_url.clone(),
                "restoreKey": restore_key
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert_eq!(get_data(&resp)["lifecycle"]["restoreStatus"], "loaded");

        let resp = execute_command(
            &json!({ "id": "11", "action": "evaluate", "script": "document.body.innerText" }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert!(
            get_data(&resp)["result"]
                .as_str()
                .unwrap_or_default()
                .contains("Welcome back"),
            "reopened session should keep cookie login on homepage: {}",
            resp
        );

        let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.previous", path));
    cleanup_restore_state_files(&restore_key);
}

#[tokio::test]
#[ignore]
async fn e2e_restore_validation_failure_does_not_overwrite_state() {
    let restore_key = format!(
        "e2e-restore-validation-{}",
        &uuid::Uuid::new_v4().to_string()[..8]
    );

    {
        let mut state = DaemonState::new();
        let resp = execute_command(
            &json!({
                "id": "1",
                "action": "navigate",
                "url": "https://example.com",
                "restoreKey": restore_key
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({
                "id": "2",
                "action": "cookies_set",
                "name": "restore_validation_test",
                "value": "known_good",
                "domain": ".example.com",
                "path": "/",
                "expires": 2000000000
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(&json!({ "id": "3", "action": "close" }), &mut state).await;
        assert_success(&resp);
        assert_eq!(get_data(&resp)["saveStatus"], "saved");
    }

    let path = super::state::find_auto_state_file(&restore_key)
        .expect("first close should create restore state");
    let before = std::fs::read_to_string(&path).expect("state file should be readable");

    {
        let mut state = DaemonState::new();
        let resp = execute_command(
            &json!({
                "id": "10",
                "action": "navigate",
                "url": "https://example.com",
                "restoreKey": restore_key,
                "restoreCheckText": "text-that-does-not-exist-in-the-page"
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);
        assert_eq!(
            get_data(&resp)["lifecycle"]["restoreStatus"],
            "loaded_but_invalid"
        );

        let resp = execute_command(&json!({ "id": "11", "action": "close" }), &mut state).await;
        assert_success(&resp);
        assert_eq!(get_data(&resp)["saveStatus"], "skipped_restore_failed");
    }

    let after = std::fs::read_to_string(&path).expect("state file should still be readable");
    assert_eq!(
        before, after,
        "failed restore validation must not overwrite previous state"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.previous", path));
}

#[tokio::test]
#[ignore]
async fn e2e_restore_key_switch_reloads_instead_of_reusing_live_browser() {
    let restore_key_a = format!(
        "e2e-restore-switch-a-{}",
        &uuid::Uuid::new_v4().to_string()[..8]
    );
    let restore_key_b = format!(
        "e2e-restore-switch-b-{}",
        &uuid::Uuid::new_v4().to_string()[..8]
    );
    let cookie_name = "restore_switch_test";
    let env = EnvGuard::new(&[
        "AGENT_BROWSER_SESSION_NAME",
        "AGENT_BROWSER_RESTORE_SAVE",
        "AGENT_BROWSER_STATE",
        "AGENT_BROWSER_ENCRYPTION_KEY",
    ]);
    env.remove("AGENT_BROWSER_SESSION_NAME");
    env.remove("AGENT_BROWSER_RESTORE_SAVE");
    env.remove("AGENT_BROWSER_STATE");
    env.remove("AGENT_BROWSER_ENCRYPTION_KEY");

    create_restore_state_with_cookie(&restore_key_a, cookie_name, "value-a").await;
    create_restore_state_with_cookie(&restore_key_b, cookie_name, "value-b").await;

    let path_a = super::state::find_auto_state_file(&restore_key_a)
        .expect("first restore key should have saved state");
    let path_b = super::state::find_auto_state_file(&restore_key_b)
        .expect("second restore key should have saved state");

    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({
            "id": "10",
            "action": "navigate",
            "url": "https://example.com",
            "restoreKey": restore_key_a
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "11", "action": "cookies_get" }), &mut state).await;
    assert_success(&resp);
    let cookies = get_data(&resp)["cookies"].as_array().unwrap();
    assert!(
        cookies
            .iter()
            .any(|c| c["name"] == cookie_name && c["value"] == "value-a"),
        "first restore key should load value-a: {:?}",
        cookies
    );

    let resp = execute_command(
        &json!({
            "id": "20",
            "action": "navigate",
            "url": "https://example.com",
            "restoreKey": restore_key_b
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["lifecycle"]["relaunchedBrowser"], true);

    let resp = execute_command(&json!({ "id": "21", "action": "cookies_get" }), &mut state).await;
    assert_success(&resp);
    let cookies = get_data(&resp)["cookies"].as_array().unwrap();
    assert!(
        cookies
            .iter()
            .any(|c| c["name"] == cookie_name && c["value"] == "value-b"),
        "switching restore keys should load value-b, not reuse value-a: {:?}",
        cookies
    );

    let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    assert_success(&resp);

    let saved_b = std::fs::read_to_string(&path_b).expect("state file should remain readable");
    assert!(saved_b.contains("value-b"));
    assert!(!saved_b.contains("value-a"));

    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(format!("{}.previous", path_a));
    let _ = std::fs::remove_file(&path_b);
    let _ = std::fs::remove_file(format!("{}.previous", path_b));
}

/// Verify that explicit `state_load` restores cookies into an existing
/// session (baseline sanity check — this path is known to work).
#[tokio::test]
#[ignore]
async fn e2e_explicit_state_load_restores_cookies() {
    let state_path = std::env::temp_dir()
        .join(format!(
            "agent-browser-e2e-explicit-load-{}.json",
            uuid::Uuid::new_v4()
        ))
        .to_string_lossy()
        .to_string();

    // Session 1: set cookie, save state
    {
        let mut state = DaemonState::new();

        let resp = execute_command(
            &json!({ "id": "1", "action": "launch", "headless": true }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({
                "id": "3",
                "action": "cookies_set",
                "name": "explicit_load_test",
                "value": "manually_loaded",
                "domain": ".example.com",
                "path": "/",
                "expires": 2000000000
            }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "4", "action": "state_save", "path": &state_path }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(&json!({ "id": "5", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    // Session 2: launch clean, then explicitly load state
    {
        let mut state = DaemonState::new();

        let resp = execute_command(
            &json!({ "id": "10", "action": "launch", "headless": true }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "11", "action": "state_load", "path": &state_path }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp = execute_command(
            &json!({ "id": "12", "action": "navigate", "url": "https://example.com" }),
            &mut state,
        )
        .await;
        assert_success(&resp);

        let resp =
            execute_command(&json!({ "id": "13", "action": "cookies_get" }), &mut state).await;
        assert_success(&resp);
        let cookies = get_data(&resp)["cookies"].as_array().unwrap();
        let found = cookies
            .iter()
            .any(|c| c["name"] == "explicit_load_test" && c["value"] == "manually_loaded");
        assert!(
            found,
            "Cookie should be present after explicit state_load. Cookies found: {:?}",
            cookies
                .iter()
                .map(|c| c["name"].as_str().unwrap_or("?"))
                .collect::<Vec<_>>()
        );

        let resp = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
        assert_success(&resp);
    }

    let _ = std::fs::remove_file(&state_path);
}

// === React / Web Vitals primitives ===

const REACT_FIXTURE_HTML: &str = r#"<!doctype html>
<html>
  <head><title>React fixture</title></head>
  <body>
    <div id="root"></div>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script>
      const { useState, createElement: h } = React;
      function Counter({ label }) {
        const [n, setN] = useState(0);
        return h("button", { onClick: () => setN(n + 1) }, label + ": " + n);
      }
      function App() {
        return h("div", {}, [
          h("h1", { key: "t" }, "Hello"),
          h(Counter, { key: "c1", label: "A" }),
          h(Counter, { key: "c2", label: "B" }),
        ]);
      }
      ReactDOM.createRoot(document.getElementById("root")).render(h(App));
    </script>
  </body>
</html>
"#;

fn react_fixture_url() -> String {
    format!(
        "data:text/html;base64,{}",
        STANDARD.encode(REACT_FIXTURE_HTML)
    )
}

#[tokio::test]
#[ignore]
async fn e2e_react_tree_errors_without_hook() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Without --enable react-devtools, the hook isn't installed and the
    // command should error.
    let resp = execute_command(&json!({ "id": "3", "action": "react_tree" }), &mut state).await;
    let err = resp
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(
        err.contains("React DevTools") || err.contains("renderer"),
        "Expected hook-missing error, got: {:?}",
        resp
    );

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

#[tokio::test]
#[ignore]
async fn e2e_react_tree_with_enable_hook() {
    let guard = EnvGuard::new(&["AGENT_BROWSER_ENABLE"]);
    guard.set("AGENT_BROWSER_ENABLE", "react-devtools");
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": &react_fixture_url() }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Give React a moment to boot and register with the hook.
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

    let resp = execute_command(&json!({ "id": "3", "action": "react_tree" }), &mut state).await;
    assert_success(&resp);
    let tree = get_data(&resp)
        .get("tree")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    assert!(
        tree.contains("App"),
        "Expected tree to contain 'App': {}",
        tree
    );
    assert!(
        tree.contains("Counter"),
        "Expected tree to contain 'Counter': {}",
        tree
    );

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

#[tokio::test]
#[ignore]
async fn e2e_relaunch_when_enable_changes_installs_react_hook() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": &react_fixture_url() }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "3", "action": "react_tree" }), &mut state).await;
    assert!(
        resp.get("success").and_then(|v| v.as_bool()) == Some(false),
        "React tree should fail before react-devtools is enabled: {}",
        resp
    );

    let resp = execute_command(
        &json!({
            "id": "4",
            "action": "launch",
            "headless": true,
            "enable": ["react-devtools"]
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(
        get_data(&resp).get("reused").is_none(),
        "changed enable list must relaunch, not reuse: {}",
        resp
    );
    assert_eq!(
        get_data(&resp)["lifecycle"]["relaunchedBrowser"],
        true,
        "lifecycle should report browser relaunch"
    );

    let resp = execute_command(
        &json!({ "id": "5", "action": "navigate", "url": &react_fixture_url() }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

    let resp = execute_command(&json!({ "id": "6", "action": "react_tree" }), &mut state).await;
    assert_success(&resp);
    let tree = get_data(&resp)
        .get("tree")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(
        tree.contains("App"),
        "Expected tree to contain App: {}",
        tree
    );

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

#[tokio::test]
#[ignore]
async fn e2e_vitals_reports_metrics() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": &react_fixture_url() }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "3", "action": "vitals" }), &mut state).await;
    assert_success(&resp);
    let data = get_data(&resp);
    assert!(data.get("url").and_then(|v| v.as_str()).is_some());
    assert!(data.get("ttfb").is_some());
    assert!(data.get("cls").and_then(|v| v.get("score")).is_some());
    assert!(data.get("phases").and_then(|v| v.as_array()).is_some());
    assert!(data
        .get("hydratedComponents")
        .and_then(|v| v.as_array())
        .is_some());

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

async fn start_a11y_frame_server() -> (u16, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    let handle = tokio::spawn(async move {
        for _ in 0..100 {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");

                let (status, content_type, body) = match path {
                    "/top" => (
                        "200 OK",
                        "text/html",
                        format!(
                            r#"<!doctype html><html lang="en"><head><title>Top</title></head>
<body><main aria-label="Top"><h1>Top</h1>
<iframe id="outer" title="Outer" src="http://127.0.0.1:{port}/outer"></iframe>
</main></body></html>"#
                        ),
                    ),
                    "/outer" => (
                        "200 OK",
                        "text/html",
                        r#"<!doctype html><html lang="en"><head><title>Outer</title></head>
<body><main aria-label="Outer"><h1>Outer</h1><img id="outer-image" src="/missing-outer.png">
<iframe id="inner" title="Inner" src="/inner"></iframe></main></body></html>"#
                            .to_string(),
                    ),
                    "/inner" => (
                        "200 OK",
                        "text/html",
                        r#"<!doctype html><html lang="en"><head><title>Inner</title></head>
<body><main aria-label="Inner"><h1>Inner</h1><img id="inner-image" src="/missing-inner.png"></main></body></html>"#
                            .to_string(),
                    ),
                    "/siblings" => (
                        "200 OK",
                        "text/html",
                        format!(
                            r#"<!doctype html><html lang="en"><head><title>Siblings</title></head>
<body><main><h1>Siblings</h1>
<iframe id="first-frame" title="First" src="http://127.0.0.1:{port}/first-frame"></iframe>
<iframe id="second-frame" title="Second" src="http://127.0.0.1:{port}/second-frame"></iframe>
</main></body></html>"#
                        ),
                    ),
                    "/first-frame" => (
                        "200 OK",
                        "text/html",
                        r#"<!doctype html><html lang="en"><head><title>First</title></head>
<body><main><h1>First</h1><img id="first-image" src="/missing-first.png"></main></body></html>"#
                            .to_string(),
                    ),
                    "/second-frame" => (
                        "200 OK",
                        "text/html",
                        r#"<!doctype html><html lang="en"><head><title>Second</title></head>
<body><main><h1>Second</h1><img id="second-image" src="/missing-second.png"></main></body></html>"#
                            .to_string(),
                    ),
                    "/background" => (
                        "200 OK",
                        "text/html",
                        format!(
                            r#"<!doctype html><html lang="en"><head><title>Background</title></head>
<body><main><h1>Background</h1>
<iframe id="background-frame" title="Background frame" src="http://127.0.0.1:{port}/background-frame"></iframe>
</main></body></html>"#
                        ),
                    ),
                    "/background-frame" => (
                        "200 OK",
                        "text/html",
                        r#"<!doctype html><html lang="en"><head><title>Background frame</title></head>
<body><main><h1>Background frame</h1><img id="background-image" src="/missing-background.png"></main></body></html>"#
                            .to_string(),
                    ),
                    _ => ("404 Not Found", "text/plain", "not found".to_string()),
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    (port, handle)
}

#[tokio::test]
#[ignore]
async fn e2e_recording_cursor_uses_page_coordinates_for_oopif() {
    let (port, server) = start_a11y_frame_server().await;
    let mut state = DaemonState::new();
    assert_success(
        &execute_command(&json!({"action": "launch", "headless": true}), &mut state).await,
    );
    assert_success(
        &execute_command(
            &json!({"action": "navigate", "url": format!("http://localhost:{port}/top")}),
            &mut state,
        )
        .await,
    );
    assert_success(&execute_command(&json!({"action": "evaluate", "script": "document.getElementById('outer').style.cssText = 'position:absolute;left:240px;top:160px;width:400px;height:300px;border:10px solid black'"}), &mut state).await);
    let child_session = state
        .iframe_sessions
        .values()
        .next()
        .expect("fixture must use an OOPIF")
        .clone();
    state.browser.as_ref().unwrap().client.send_command("Runtime.evaluate", Some(json!({
        "expression": "document.body.innerHTML = '<button style=\"position:absolute;left:20px;top:30px;width:80px;height:40px\">Cursor target</button>'"
    })), Some(&child_session)).await.unwrap();
    let snapshot = execute_command(&json!({"action": "snapshot"}), &mut state).await;
    assert_success(&snapshot);
    let reference = get_data(&snapshot)["refs"]
        .as_object()
        .unwrap()
        .iter()
        .find(|(_, entry)| entry["name"] == "Cursor target")
        .unwrap()
        .0
        .clone();
    // Verify the input samples retain top-level coordinates for frame actions.
    super::recording::recording_start(
        &mut state.recording_state,
        "unused.webm",
        super::recording::RecordingOptions {
            cursor: true,
            ..Default::default()
        },
    )
    .unwrap();
    for action in ["hover", "click", "dblclick"] {
        assert_success(&execute_command(&json!({"action": action, "selector": format!("@{reference}"), "inputMode": "human"}), &mut state).await);
        let cursor = state
            .recording_state
            .shared_cursor
            .lock()
            .unwrap()
            .at(f64::INFINITY);
        assert!(cursor.visible);
        assert!(
            (cursor.x - 310.0).abs() < 1.0 && (cursor.y - 220.0).abs() < 1.0,
            "{action}: cursor should be at page (310, 220), got {cursor:?}"
        );
        assert!((state.mouse_state.x - cursor.x).abs() < 1.0);
        assert!((state.mouse_state.y - cursor.y).abs() < 1.0);
    }
    state.recording_state.active = false;
    state.browser.as_ref().unwrap().client.send_command("Runtime.evaluate", Some(json!({
        "expression": "document.body.style.background='#303030'; const button = document.querySelector('button'); button.style.background='#303030'; button.style.border='0'; button.textContent=''"
    })), Some(&child_session)).await.unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("iframe-cursor.mp4");
    assert_success(
        &execute_command(
            &json!({"action":"recording_start", "path":path,"cursor":true,"fps":60}),
            &mut state,
        )
        .await,
    );
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    assert_success(
        &execute_command(
            &json!({"action":"hover", "selector":format!("@{reference}"), "inputMode":"human"}),
            &mut state,
        )
        .await,
    );
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let stopped = execute_command(&json!({"action":"recording_stop"}), &mut state).await;
    assert_success(&execute_command(&json!({"action": "close"}), &mut state).await);
    server.abort();
    assert_success(&stopped);
    let output = tokio::process::Command::new("ffmpeg")
        .args(["-v", "error", "-sseof", "-0.1", "-i"])
        .arg(path)
        .args([
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "pipe:1",
        ])
        .output()
        .await
        .unwrap();
    assert!(output.status.success());
    let image = image::load_from_memory(&output.stdout).unwrap().to_rgb8();
    assert!(
        image
            .get_pixel(314, 224)
            .0
            .iter()
            .all(|channel| *channel > 180),
        "the cursor should be visible inside the out-of-process iframe"
    );
}

#[tokio::test]
#[ignore]
async fn e2e_a11y_uses_vendored_engine_and_preserves_shadow_targets() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = r#"<!doctype html>
<html lang="en">
<head><title>Accessibility audit fixture</title></head>
<body>
  <main>
    <h1>Accessibility audit fixture</h1>
    <img id="light-image" src="missing.png">
    <div id="shadow-host"></div>
    <iframe id="audit-frame" title="Audit frame" tabindex="-1" srcdoc="
      <!doctype html><html lang='en'><head><title>Frame</title></head>
      <body><main><h1>Frame</h1><a href='https://example.com'>Frame link</a><img id='frame-image' src='missing.png'></main>
      <script>
        window.frameAxe = { version: 'frame-spoofed' };
        window.frameAxeSetterCalls = 0;
        Object.defineProperty(window, 'axe', {
          configurable: false,
          get() { return window.frameAxe; },
          set() {
            window.frameAxeSetterCalls += 1;
            throw new Error('frame axe setter must not run');
          }
        });
      </script></body></html>
    "></iframe>
  </main>
  <script>
    window.pageAxe = {
      version: 'spoofed',
      run: () => Promise.resolve({
        url: 'spoofed',
        testEngine: { version: 'spoofed' },
        violations: [],
        incomplete: [],
        passes: [],
        inapplicable: []
      })
    };
    window.axeSetterCalls = 0;
    Object.defineProperty(window, 'axe', {
      configurable: false,
      get() { return window.pageAxe; },
      set() {
        window.axeSetterCalls += 1;
        throw new Error('page axe setter must not run');
      }
    });
    window.amdCalls = 0;
    window.define = () => { window.amdCalls += 1; };
    window.define.amd = {};
    document.getElementById('shadow-host').attachShadow({ mode: 'open' }).innerHTML =
      '<img id="shadow-image" src="missing.png">';
  </script>
</body>
</html>"#;
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "3", "action": "a11y" }), &mut state).await;
    assert_success(&resp);
    let data = get_data(&resp);
    assert_eq!(data["axeVersion"], "4.12.1");
    let image_alt = data["violations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|violation| violation["id"] == "image-alt")
        .expect("vendored axe should report missing image alternatives");
    assert_eq!(image_alt["nodeCount"], 3);
    let nodes = image_alt["nodes"].as_array().unwrap();
    assert!(nodes
        .iter()
        .any(|node| node["target"] == json!(["#light-image"])));
    assert!(nodes
        .iter()
        .any(|node| node["target"] == json!([["#shadow-host", "#shadow-image"]])));
    assert!(nodes
        .iter()
        .any(|node| node["target"] == json!(["#audit-frame", "#frame-image"])));
    let frame_focusable_content = data["violations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|violation| violation["id"] == "frame-focusable-content")
        .expect("audit should preserve the child context for non-focusable frames");
    assert!(frame_focusable_content["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|node| node["target"] == json!(["#audit-frame", "html"])));

    let resp = execute_command(
        &json!({ "id": "3-selector", "action": "a11y", "selector": "#shadow-host" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let scoped_image_alt = get_data(&resp)["violations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|violation| violation["id"] == "image-alt")
        .expect("scoped audit should report the shadow image");
    assert_eq!(scoped_image_alt["nodeCount"], 1);

    let resp = execute_command(
        &json!({
            "id": "4",
            "action": "evaluate",
            "script": "[window.axe.version, document.querySelector('#audit-frame').contentWindow.axe.version, window.axeSetterCalls, document.querySelector('#audit-frame').contentWindow.frameAxeSetterCalls, window.amdCalls]"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["result"],
        json!(["spoofed", "frame-spoofed", 0, 0, 0])
    );

    state
        .ref_map
        .add("e999".to_string(), Some(999), "button", "stale", None);
    state.active_frame_id = Some("stale-frame".to_string());
    state
        .iframe_sessions
        .insert("stale-frame".to_string(), "stale-session".to_string());
    let fresh_url = format!(
        "data:text/html;base64,{}",
        STANDARD.encode(
            "<!doctype html><html lang='en'><head><title>Fresh audit</title></head><body><main><h1>Fresh audit</h1></main></body></html>"
        )
    );
    let resp = execute_command(
        &json!({ "id": "5", "action": "a11y", "url": fresh_url }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(state.ref_map.get("e999").is_none());
    assert!(state.active_frame_id.is_none());
    assert!(!state.iframe_sessions.contains_key("stale-frame"));

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

#[tokio::test]
#[ignore]
async fn e2e_a11y_preserves_nested_frame_sessions_across_tab_switches() {
    let (port, server) = start_a11y_frame_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": format!("http://localhost:{port}/top")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert!(
        !state.iframe_sessions.is_empty(),
        "cross-origin frame should have an attached target session"
    );
    let top_iframe_sessions = state.active_iframe_sessions.clone();
    assert!(
        !top_iframe_sessions.is_empty(),
        "active frame sessions should include the top tab's cross-origin frame"
    );

    let assert_frame_violations = |resp: &Value| {
        assert_success(resp);
        let image_alt = get_data(resp)["violations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|violation| violation["id"] == "image-alt")
            .unwrap_or_else(|| {
                panic!(
                    "audit should include images in nested cross-origin frames: {}",
                    serde_json::to_string_pretty(resp).unwrap_or_default()
                )
            });
        assert_eq!(image_alt["nodeCount"], 2);
        let nodes = image_alt["nodes"].as_array().unwrap();
        assert!(nodes
            .iter()
            .any(|node| node["target"] == json!(["#outer", "#outer-image"])));
        assert!(nodes
            .iter()
            .any(|node| node["target"] == json!(["#outer", "#inner", "#inner-image"])));
    };

    let resp = execute_command(&json!({ "id": "3", "action": "a11y" }), &mut state).await;
    assert_frame_violations(&resp);

    let resp = execute_command(
        &json!({ "id": "3-selector", "action": "a11y", "selector": "main" }),
        &mut state,
    )
    .await;
    assert_frame_violations(&resp);

    // Simulate a popup created outside the tab commands. Target lifecycle
    // events must move active iframe scoping to the popup and back when it is
    // externally closed.
    let browser_client = state.browser.as_ref().unwrap().client.clone();
    let created = browser_client
        .send_command(
            "Target.createTarget",
            Some(json!({
                "url": format!("http://localhost:{port}/background")
            })),
            None,
        )
        .await
        .unwrap();
    let external_target_id = created["targetId"].as_str().unwrap().to_string();
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let resp = execute_command(
        &json!({ "id": "external-open", "action": "tab_list" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let resp = execute_command(
        &json!({ "id": "external-loaded", "action": "tab_list" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let external_iframe_sessions = state.active_iframe_sessions.clone();
    assert!(!external_iframe_sessions.is_empty());
    assert!(top_iframe_sessions.is_disjoint(&external_iframe_sessions));

    browser_client
        .send_command(
            "Target.closeTarget",
            Some(json!({ "targetId": external_target_id })),
            None,
        )
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let resp = execute_command(
        &json!({ "id": "external-close", "action": "tab_list" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(state.active_iframe_sessions, top_iframe_sessions);

    let resp = execute_command(
        &json!({
            "id": "4",
            "action": "tab_new",
            "url": format!("http://localhost:{port}/background")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let background_iframe_sessions = state.active_iframe_sessions.clone();
    assert!(
        !background_iframe_sessions.is_empty(),
        "active frame sessions should follow the newly opened tab"
    );
    assert!(top_iframe_sessions.is_disjoint(&background_iframe_sessions));
    let resp = execute_command(
        &json!({ "id": "5", "action": "tab_switch", "tabId": "t1" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(state.active_iframe_sessions, top_iframe_sessions);

    let resp = execute_command(&json!({ "id": "6", "action": "a11y" }), &mut state).await;
    assert_frame_violations(&resp);

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_a11y_preserves_sibling_frame_dom_order() {
    let (port, server) = start_a11y_frame_server().await;
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "2",
            "action": "navigate",
            "url": format!("http://localhost:{port}/siblings")
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(&json!({ "id": "3", "action": "a11y" }), &mut state).await;
    assert_success(&resp);
    let image_alt = get_data(&resp)["violations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|violation| violation["id"] == "image-alt")
        .expect("audit should report both sibling frame images");
    assert_eq!(image_alt["nodeCount"], 2);
    let targets: Vec<_> = image_alt["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|node| node["target"].clone())
        .collect();
    assert_eq!(
        targets,
        vec![
            json!(["#first-frame", "#first-image"]),
            json!(["#second-frame", "#second-image"]),
        ]
    );

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
    server.abort();
}

#[tokio::test]
#[ignore]
async fn e2e_pushstate_changes_url() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com/" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "pushstate", "url": "/newpath" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let url = get_data(&resp)
        .get("url")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(
        url.ends_with("/newpath"),
        "Expected pushstate URL to end with /newpath, got: {}",
        url
    );

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

#[tokio::test]
#[ignore]
async fn e2e_removeinitscript_roundtrip() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": "https://example.com" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({
            "id": "3",
            "action": "addinitscript",
            "script": "window.__AB_ROUNDTRIP__ = 1;"
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let identifier = get_data(&resp)["identifier"]
        .as_str()
        .expect("addinitscript should return an identifier")
        .to_string();
    assert!(!identifier.is_empty());

    let resp = execute_command(
        &json!({ "id": "4", "action": "removeinitscript", "identifier": identifier }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["removed"], true);

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

// ---------------------------------------------------------------------------
// Session-to-tab binding (--pin-tab): two sessions sharing one Chrome over
// CDP must not hijack each other's tabs. A restarted daemon re-binds to its
// persisted target instead of adopting the most recently active tab, and a
// pinned session whose tab is closed gets a tab_gone error instead of
// silently acting on another session's tab.
// ---------------------------------------------------------------------------

/// Environment guard for binding tests: isolated socket dir (so binding
/// files cannot leak between test runs) plus per-session env vars.
fn binding_test_env() -> (EnvGuard<'static>, tempfile::TempDir) {
    let guard = EnvGuard::new(&[
        "AGENT_BROWSER_SOCKET_DIR",
        "XDG_RUNTIME_DIR",
        "AGENT_BROWSER_NAMESPACE",
        "AGENT_BROWSER_SESSION",
        "AGENT_BROWSER_PIN_TAB",
    ]);
    let dir = tempfile::tempdir().unwrap();
    guard.set("AGENT_BROWSER_SOCKET_DIR", dir.path().to_str().unwrap());
    guard.remove("XDG_RUNTIME_DIR");
    guard.remove("AGENT_BROWSER_NAMESPACE");
    guard.remove("AGENT_BROWSER_PIN_TAB");
    (guard, dir)
}

/// Launch a host Chrome and return (host_state, ws_url) for other sessions
/// to attach to over CDP.
async fn launch_binding_host(guard: &EnvGuard<'static>) -> (DaemonState, String) {
    guard.set("AGENT_BROWSER_SESSION", "e2e-bind-host");
    let mut host = DaemonState::new();
    let resp = execute_command(
        &json!({
            "id": "host-1",
            "action": "launch",
            "headless": true,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"]
        }),
        &mut host,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(&json!({ "id": "host-2", "action": "cdp_url" }), &mut host).await;
    assert_success(&resp);
    let ws_url = get_data(&resp)["cdpUrl"]
        .as_str()
        .expect("cdpUrl should be a string")
        .to_string();
    (host, ws_url)
}

/// Create a pinned session attached to the shared Chrome and navigate its
/// bound tab to `url`. Returns the session's state.
async fn attach_pinned_session(
    guard: &EnvGuard<'static>,
    session: &str,
    ws_url: &str,
    url: &str,
) -> DaemonState {
    guard.set("AGENT_BROWSER_SESSION", session);
    let mut state = DaemonState::new();
    let resp = execute_command(
        &json!({
            "id": format!("{}-launch", session),
            "action": "launch",
            "cdpUrl": ws_url,
            "pinTab": true
        }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": format!("{}-nav", session), "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    state
}

async fn current_url(state: &mut DaemonState, id: &str) -> Value {
    execute_command(&json!({ "id": id, "action": "url" }), state).await
}

fn load_binding(session: &str, expect: &str) -> super::tab_binding::TabBinding {
    super::tab_binding::load(session)
        .expect("binding file should be readable")
        .expect(expect)
}

#[tokio::test]
#[ignore]
async fn e2e_pin_tab_rebinds_after_daemon_restart() {
    let (guard, _dir) = binding_test_env();
    let (mut host, ws_url) = launch_binding_host(&guard).await;

    let url_a = "data:text/html,session-a-page";
    let url_b = "data:text/html,session-b-page";

    // Session A binds its own tab (pin mode creates a fresh tab on attach).
    let state_a = attach_pinned_session(&guard, "e2e-bind-a", &ws_url, url_a).await;
    let binding_a = load_binding("e2e-bind-a", "session A binding should persist");
    assert!(binding_a.pinned, "binding should record pinned=true");
    assert_eq!(
        binding_a.url, "",
        "opaque URL payloads must not be persisted in diagnostic state"
    );

    // Session B binds its own tab and navigates last, so B's tab is the most
    // recently active one (the tab a naive re-attach would adopt).
    let mut state_b = attach_pinned_session(&guard, "e2e-bind-b", &ws_url, url_b).await;
    let binding_b = load_binding("e2e-bind-b", "session B binding should persist");
    assert_ne!(
        binding_a.target_id, binding_b.target_id,
        "sessions must bind distinct tabs"
    );

    // Simulate session A's daemon dying (idle timeout, crash, kill): drop the
    // state without a clean close. The binding file survives.
    drop(state_a);
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    // A restarted daemon for session A must re-bind to A's original tab by
    // targetId, not adopt B's (most recently active) tab.
    guard.set("AGENT_BROWSER_SESSION", "e2e-bind-a");
    let mut state_a2 = DaemonState::new();
    assert!(
        state_a2.pin_tab,
        "pin-tab should be sticky via the persisted binding"
    );
    let resp = execute_command(
        &json!({ "id": "a2-launch", "action": "launch", "cdpUrl": ws_url }),
        &mut state_a2,
    )
    .await;
    assert_success(&resp);

    let resp = current_url(&mut state_a2, "a2-url").await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["url"],
        url_a,
        "restarted session A must operate on A's original tab"
    );
    let binding_a2 = load_binding("e2e-bind-a", "binding should still exist");
    assert_eq!(
        binding_a2.target_id, binding_a.target_id,
        "re-attach must keep the original bound target"
    );
    assert_eq!(
        binding_a2.url, "",
        "re-attach must not reintroduce the opaque URL payload"
    );

    // Lifecycle-dependent commands must work on the restored tab: attach
    // only enables the CDP domains on the first target, so the restore path
    // must enable them on the bound tab too or navigate hangs waiting for
    // Page lifecycle events (timeout guards against the hang regression).
    let url_a2 = "data:text/html,session-a-page-2";
    let resp = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        execute_command(
            &json!({ "id": "a2-nav", "action": "navigate", "url": url_a2 }),
            &mut state_a2,
        ),
    )
    .await
    .expect("navigate on the restored tab must not hang");
    assert_success(&resp);
    assert_eq!(get_data(&resp)["targetId"], binding_a.target_id);

    // Session B is unaffected.
    let resp = current_url(&mut state_b, "b-url").await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["url"], url_b);

    let resp = execute_command(&json!({ "id": "host-99", "action": "close" }), &mut host).await;
    assert_success(&resp);
}

// Real-Chrome smoke test: a foreign tab opened via `window.open` inside the
// pinned session's own page must never steal the active tab or overwrite the
// pin binding. NOTE: this does NOT specifically guard finding #3 (the
// `Target.attachedToTarget`-before-`Target.targetCreated` race) — empirically
// in this environment Chrome always delivers `Target.targetCreated` first for
// a `window.open()` popup, so this only exercises the `new_targets` drain
// path in actions.rs (~line 1082), which was never the buggy branch. The
// deterministic, race-independent regression coverage for finding #3 itself
// lives at the unit level: `BrowserManager::register_discovered_page` (the
// single decision point both drain handlers call) is exercised directly by
// `test_register_discovered_page_untracked_target_does_not_steal_pinned_tab`
// in browser.rs, which fails if that function's internal `add_page` vs.
// `add_page_without_activation` choice regresses. This e2e test is kept as a
// general non-regression smoke check against real Chrome, not as #3 proof.
#[tokio::test]
#[ignore]
async fn e2e_auto_attached_foreign_tab_does_not_steal_pinned_tab() {
    let (guard, _dir) = binding_test_env();
    let (mut host, ws_url) = launch_binding_host(&guard).await;

    let url_a = "data:text/html,pinned-session-a";
    let mut state_a = attach_pinned_session(&guard, "e2e-foreign-a", &ws_url, url_a).await;
    let binding_before = load_binding("e2e-foreign-a", "session A binding should persist");

    // Open a foreign tab in the shared browser. This is not an agent command
    // (`tab new`), it is a plain popup — the same shape as a human opening a
    // tab or a page calling `window.open`.
    let resp = execute_command(
        &json!({
            "id": "foreign-open",
            "action": "evaluate",
            "script": "window.open('data:text/html,foreign-popup'); 'opened'"
        }),
        &mut state_a,
    )
    .await;
    assert_success(&resp);

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let resp = current_url(&mut state_a, "a-url-after-foreign").await;
    assert_success(&resp);

    let mgr = state_a.browser.as_ref().expect("browser should be running");
    let page_count_before = 1; // the pinned session's own bound tab
    assert!(
        mgr.page_count() > page_count_before,
        "the foreign popup must still register in tab list (got {} pages)",
        mgr.page_count()
    );
    assert_eq!(
        get_data(&resp)["url"],
        url_a,
        "the pinned session's active tab must not be stolen by the foreign popup"
    );
    assert_eq!(
        mgr.bound_target_id(),
        Some(binding_before.target_id.as_str()),
        "the pin binding must not be overwritten by the foreign popup"
    );

    let resp = execute_command(&json!({ "id": "host-99", "action": "close" }), &mut host).await;
    assert_success(&resp);
}

// SCRATCH (finding #3): explicit agent commands must still end up active
// even under the same auto-attach shared browser, proving the fix does not
// regress `tab new`.
#[tokio::test]
#[ignore]
async fn e2e_tab_new_still_activates_under_auto_attach() {
    let (guard, _dir) = binding_test_env();
    let (mut host, ws_url) = launch_binding_host(&guard).await;

    let url_a = "data:text/html,pinned-session-a";
    let mut state_a = attach_pinned_session(&guard, "e2e-tabnew-a", &ws_url, url_a).await;

    let resp = execute_command(
        &json!({
            "id": "tab-new",
            "action": "tab_new",
            "url": "data:text/html,agent-opened-tab"
        }),
        &mut state_a,
    )
    .await;
    assert_success(&resp);

    let resp = current_url(&mut state_a, "a-url-after-tab-new").await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["url"],
        "data:text/html,agent-opened-tab",
        "an explicit `tab new` must activate the new tab it created"
    );

    let resp = execute_command(&json!({ "id": "host-99", "action": "close" }), &mut host).await;
    assert_success(&resp);
}

#[tokio::test]
#[ignore]
async fn e2e_pin_tab_gone_error_and_recovery() {
    let (guard, _dir) = binding_test_env();
    let (mut host, ws_url) = launch_binding_host(&guard).await;

    let url_a = "data:text/html,gone-session-a";
    let url_b = "data:text/html,gone-session-b";

    let mut state_a = attach_pinned_session(&guard, "e2e-gone-a", &ws_url, url_a).await;
    let binding_a = load_binding("e2e-gone-a", "session A binding should persist");
    let mut state_b = attach_pinned_session(&guard, "e2e-gone-b", &ws_url, url_b).await;

    // Session B closes A's tab by targetId (targetIds are accepted anywhere a
    // tab ref is accepted and are stable across daemons).
    let resp = execute_command(
        &json!({ "id": "b-close-a", "action": "tab_close", "tabId": binding_a.target_id }),
        &mut state_b,
    )
    .await;
    assert_success(&resp);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Session A must now fail loudly with a machine-readable tab_gone error
    // instead of silently adopting a neighboring tab.
    let resp = current_url(&mut state_a, "a-url-gone").await;
    assert_eq!(resp["success"], false);
    assert_eq!(
        resp["code"],
        "tab_gone",
        "response should carry code=tab_gone, got: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );
    let err = resp["error"].as_str().unwrap_or("");
    assert!(
        err.starts_with(super::browser::TAB_GONE_PREFIX),
        "error should start with the tab_gone prefix, got: {}",
        err
    );
    assert_eq!(resp["data"]["targetId"], binding_a.target_id);
    assert!(
        resp["data"].get("lastUrl").is_none(),
        "opaque URLs must not be exposed in structured diagnostics: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );

    // Recovery commands still work in the gone state: tab_list is allowed,
    // and tab_new binds a fresh tab.
    let resp = execute_command(
        &json!({ "id": "a-list", "action": "tab_list" }),
        &mut state_a,
    )
    .await;
    assert_success(&resp);

    let url_a2 = "data:text/html,recovered-session-a";
    let resp = execute_command(
        &json!({ "id": "a-new", "action": "tab_new", "url": url_a2 }),
        &mut state_a,
    )
    .await;
    assert_success(&resp);

    let resp = current_url(&mut state_a, "a-url-recovered").await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["url"], url_a2);

    let binding_a2 = load_binding("e2e-gone-a", "binding should be rewritten");
    assert_ne!(
        binding_a2.target_id, binding_a.target_id,
        "recovery must bind a new target"
    );
    assert!(binding_a2.pinned, "recovered binding stays pinned");

    // Session B keeps working on its own tab throughout.
    let resp = current_url(&mut state_b, "b-url-after").await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["url"], url_b);

    let resp = execute_command(&json!({ "id": "host-99", "action": "close" }), &mut host).await;
    assert_success(&resp);
}

/// A multiselect locator miss must surface the anchored "No element found"
/// guidance, not a raw "Evaluation error: ...". Guards the handler wiring end to
/// end: a unit test of the mapping alone stays green if the handler stops routing
/// misses through it. Force-red: drop the sentinel miss handling in
/// handle_multiselect and this assertion fails on the raw evaluate error.
#[tokio::test]
#[ignore]
async fn e2e_multiselect_miss_surfaces_anchored_error() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // about:blank has no #picker, so the selector misses.
    let resp = execute_command(
        &json!({ "id": "2", "action": "multiselect", "selector": "#picker", "values": ["a"] }),
        &mut state,
    )
    .await;
    assert_eq!(
        resp.get("success").and_then(|v| v.as_bool()),
        Some(false),
        "multiselect on a missing selector should error: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );
    let err = resp.get("error").and_then(|v| v.as_str()).unwrap_or("");
    // Assert the full anchored shape, not just the guidance suffix: the miss must
    // carry "No element found", retain the "#picker" selector detail, and end with
    // the locator-miss guidance. A generic suffix-only check would pass even if the
    // detail were dropped or a different classifier produced the guidance.
    assert!(
        err.starts_with("No element found") && err.contains("#picker"),
        "miss should keep the anchored shape and selector detail, got: {err}"
    );
    assert!(
        err.contains("Verify the selector, role, or name is correct"),
        "miss should surface the anchored locator-miss guidance, got: {err}"
    );

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

/// An earlier valid ARIA token in a multi-token role attribute is the operative
/// role, so `role="mark none"` is a mark and must not answer `find role none`.
/// Force-red: drop `mark` from the presentational VALID_ROLES set and the query
/// matches this element, so the miss assertion fails.
#[tokio::test]
#[ignore]
async fn e2e_presentational_role_respects_earlier_operative_token() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let html = "<html><body><span role='mark none'>marked</span></body></html>";
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "getbyrole", "role": "none", "subaction": "text" }),
        &mut state,
    )
    .await;
    assert_eq!(
        resp.get("success").and_then(|v| v.as_bool()),
        Some(false),
        "operative role is `mark`, so `find role none` must not match: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

/// `find role directory` must match an explicit `role="directory"` element but
/// not an ordinary list. Chrome collapses `directory` into the `list` AX role,
/// so this goes through the DOM-attribute path, not the AX tree. Force-red:
/// route `directory` back through the AX tree and the explicit element is missed
/// (its AX role is `list`), so the found-text assertion fails.
#[tokio::test]
#[ignore]
async fn e2e_find_role_directory_matches_only_explicit_attribute() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // A plain list must not be matched by `find role directory`.
    let plain = "data:text/html;base64,".to_string()
        + &STANDARD.encode("<html><body><ul><li>item</li></ul></body></html>");
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": plain }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "3", "action": "getbyrole", "role": "directory", "subaction": "text" }),
        &mut state,
    )
    .await;
    assert_eq!(
        resp.get("success").and_then(|v| v.as_bool()),
        Some(false),
        "a plain <ul> must not match `find role directory`: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );

    // An explicit role="directory" element is found.
    let explicit = "data:text/html;base64,".to_string()
        + &STANDARD.encode("<html><body><div role='directory'>DIR</div></body></html>");
    let resp = execute_command(
        &json!({ "id": "4", "action": "navigate", "url": explicit }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "5", "action": "getbyrole", "role": "directory", "subaction": "text" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["text"], "DIR");

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

/// The presentational DOM lookup must honor the selected frame, not always the
/// top document. Force-red: revert the frame dispatch in the presentational path
/// (search the top document only) and the in-frame element is missed.
#[tokio::test]
#[ignore]
async fn e2e_presentational_role_honors_selected_frame() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Top document has no role="none"; the same-origin iframe does. `srcdoc`
    // inherits the parent origin, so the child is same-origin (the path this fix
    // targets) without needing a server; a `data:` child would be cross-origin.
    let outer = "<body><iframe id='f' srcdoc=\"<div role='none'>INSIDE</div>\"></iframe></body>";
    let url = format!("data:text/html;base64,{}", STANDARD.encode(outer));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Before selecting the frame, the top document has no match.
    let resp = execute_command(
        &json!({ "id": "3", "action": "getbyrole", "role": "none", "subaction": "text" }),
        &mut state,
    )
    .await;
    assert_eq!(
        resp.get("success").and_then(|v| v.as_bool()),
        Some(false),
        "top document has no role=none: {}",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );

    // Select the frame; the presentational lookup must now find the frame element.
    let resp = execute_command(
        &json!({ "id": "4", "action": "frame", "selector": "#f" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    let resp = execute_command(
        &json!({ "id": "5", "action": "getbyrole", "role": "none", "subaction": "text" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(get_data(&resp)["text"], "INSIDE");

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

/// ARIA presentational-roles conflict resolution: role="none" on a focusable
/// element (or one with global ARIA props) is ignored, so `find role none` must
/// skip it but still match a truly presentational element. Force-red: drop the
/// conflict check and the `<button role="none">` is matched.
#[tokio::test]
#[ignore]
async fn e2e_presentational_role_respects_conflict_resolution() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // Each non-matching element triggers conflict resolution a different way:
    // native focusable, explicit tabindex=-1 (programmatically focusable), and a
    // global ARIA property. Only the last, truly presentational div must match.
    let html = concat!(
        "<body>",
        "<button role='none'>NativeFocusable</button>",
        "<div role='none' tabindex='-1'>TabindexFocusable</div>",
        "<div role='none' aria-label='named'>GlobalAria</div>",
        "<div role='none'>Plain</div>",
        "</body>"
    );
    let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    // The focusable button keeps its implicit role, so the match must be the div.
    let resp = execute_command(
        &json!({ "id": "3", "action": "getbyrole", "role": "none", "subaction": "text" }),
        &mut state,
    )
    .await;
    assert_success(&resp);
    assert_eq!(
        get_data(&resp)["text"],
        "Plain",
        "role=none on a focusable button must be ignored (conflict resolution)"
    );

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

/// `find role document` must match the page root. Chrome exposes it as
/// `RootWebArea`; without the normalization to `document` the query misses.
/// End-to-end guard for that mapping (the unit test only checks the string).
#[tokio::test]
#[ignore]
async fn e2e_find_role_document_matches_root() {
    let mut state = DaemonState::new();

    let resp = execute_command(
        &json!({ "id": "1", "action": "launch", "headless": true }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let url = format!(
        "data:text/html;base64,{}",
        STANDARD.encode("<title>T</title><body>hi</body>")
    );
    let resp = execute_command(
        &json!({ "id": "2", "action": "navigate", "url": url }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let resp = execute_command(
        &json!({ "id": "3", "action": "getbyrole", "role": "document", "subaction": "text" }),
        &mut state,
    )
    .await;
    assert_success(&resp);

    let _ = execute_command(&json!({ "id": "99", "action": "close" }), &mut state).await;
}

#[tokio::test]
#[ignore]
async fn e2e_mouse_interpolation_starts_at_last_element_interaction() {
    let mut state = DaemonState::new();
    assert_success(
        &execute_command(
            &json!({"id": "1", "action": "launch", "headless": true}),
            &mut state,
        )
        .await,
    );
    assert_success(&execute_command(&json!({"id": "2", "action": "setcontent", "html": "<button id='b' style='position:absolute;left:400px;top:300px;width:100px;height:40px'>Target</button><input id='c' type='checkbox' style='position:absolute;left:200px;top:200px'>"}), &mut state).await);
    for action in ["click", "hover", "dblclick", "check", "uncheck"] {
        let selector = if matches!(action, "check" | "uncheck") {
            "#c"
        } else {
            "#b"
        };
        assert_success(
            &execute_command(
                &json!({"id": "3", "action": action, "selector": selector}),
                &mut state,
            )
            .await,
        );
        let start = (state.mouse_state.x, state.mouse_state.y);
        assert!(start.0 >= 200.0 && start.1 >= 200.0, "{action}: {start:?}");
        assert_success(&execute_command(&json!({"id": "4", "action": "evaluate", "script": "window.moves=[];document.onmousemove=e=>moves.push([e.clientX,e.clientY]);"}), &mut state).await);
        assert_success(&execute_command(&json!({"id": "5", "action": "mousemove", "x": start.0 + 100.0, "y": start.1, "steps": 2}), &mut state).await);
        let result = execute_command(
            &json!({"id": "6", "action": "evaluate", "script": "moves"}),
            &mut state,
        )
        .await;
        assert_success(&result);
        let moves = get_data(&result)["result"].as_array().unwrap();
        assert_eq!(moves.len(), 2, "{action}: {moves:?}");
        assert!((moves[0][0].as_f64().unwrap() - (start.0 + 50.0)).abs() <= 1.0);
        assert!((moves[1][0].as_f64().unwrap() - (start.0 + 100.0)).abs() <= 1.0);
    }
    assert_success(&execute_command(&json!({"id": "99", "action": "close"}), &mut state).await);
}

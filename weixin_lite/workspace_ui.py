from __future__ import annotations

from uuid import uuid4

import streamlit as st

from .workspace_store import (STATE_KEYS, list_workspaces, load_workspace, restore_bytes,
                              save_workspace, snapshot_bytes)


def checkpoint():
    if "project_id" not in st.session_state:
        return
    try:
        save_workspace(st.session_state)
        st.session_state.pop("save_error", None)
    except (OSError, ValueError) as exc:
        st.session_state.save_error = f"本地保存失败：{exc}"


def switch_project(values, project_id):
    for key in list(st.session_state):
        if key in STATE_KEYS or key.startswith(("asset-", "editor-", "draft-", "download-paper-selector-", "selected-order-", "block-", "figure-edit-")):
            del st.session_state[key]
    st.session_state.update(values)
    st.session_state.project_id = project_id
    st.session_state.pop("project-name-widget", None)
    st.session_state.pop("_saved_digest", None)
    st.session_state.pop("prepared_exports", None)
    st.session_state.single_analysis = None
    st.session_state.selected_download_keys = []
    st.query_params["project"] = project_id


def initialize_workspace():
    if "project_id" in st.session_state:
        return
    project_id = st.query_params.get("project", "")
    if project_id:
        try:
            switch_project(load_workspace(project_id), project_id)
        except (OSError, ValueError) as exc:
            st.session_state.save_error = f"无法恢复项目：{exc}"
            project_id = ""
    if not project_id:
        st.session_state.project_id = uuid4().hex
        st.session_state.project_name = "未命名项目"
        st.query_params["project"] = st.session_state.project_id


def render_workspace_sidebar():
    with st.sidebar.expander("本地项目", expanded=False):
        st.session_state.project_name = st.text_input(
            "项目名称", value=st.session_state.get("project_name", "未命名项目"), key="project-name-widget")
        if st.button("新建项目", icon=":material/add:"):
            checkpoint()
            switch_project({}, uuid4().hex)
            st.session_state.project_name = "未命名项目"
            st.rerun()
        options = list_workspaces()
        if options:
            labels = dict(options)
            selected = st.selectbox("已保存项目", [key for key, _ in options],
                                    format_func=lambda key: f"{labels[key]} · {key[:6]}")
            if st.button("打开项目", icon=":material/folder_open:"):
                try:
                    values = load_workspace(selected)
                    checkpoint()
                    switch_project(values, selected)
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.error(str(exc))
        if st.session_state.get("papers") or st.session_state.get("articles"):
            data, _ = snapshot_bytes(st.session_state)
            st.download_button("备份项目", data, "weixin-workspace.zip", "application/zip", icon=":material/download:")
        upload = st.file_uploader("恢复项目备份", type=["zip"], key="workspace-import")
        if st.button("导入备份", disabled=upload is None):
            try:
                values = restore_bytes(upload.getvalue())
                checkpoint()
                switch_project(values, uuid4().hex)
                checkpoint()
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(str(exc))
        if st.session_state.get("save_error"):
            st.error(st.session_state.save_error)
        elif st.session_state.get("_saved_digest"):
            st.caption("已自动保存到本机")

from fastapi import FastAPI, Request
from fastapi import Form as FastapiForm, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import APIKeyHeader
from typing import Optional, List

from xue import HTML, Head, Body, Div, xue_initialize, Script
from xue.components.menubar import (
    Menubar, MenubarMenu, MenubarTrigger, MenubarContent,
    MenubarItem, MenubarSeparator
)
from xue.components import input
from xue.components import dropdown, sheet, form, button, checkbox
from xue.components.model_config_row import model_config_row
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from components.provider_table import data_table


from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

xue_initialize(tailwind=True)

from starlette.middleware.base import BaseHTTPMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestBodyLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path.startswith("/submit/"):
        # if request.method == "POST":
            body = await request.body()
            logger.info(f"Request body for {request.url.path}: {body.decode()}")

        response = await call_next(request)
        return response

from utils import load_config
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # app.state.client = httpx.AsyncClient(timeout=timeout)
    app.state.config, app.state.api_keys_db, app.state.api_list = await load_config()
    for item in app.state.api_keys_db:
        if item.get("role") == "admin":
            app.state.admin_api_key = item.get("api")
    if not hasattr(app.state, "admin_api_key"):
        if len(app.state.api_keys_db) >= 1:
            app.state.admin_api_key = app.state.api_keys_db[0].get("api")
        else:
            raise Exception("No admin API key found")

    global data
    # providers_data = app.state.config["providers"]

    # print("data", data)
    yield
    # 关闭时的代码
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)
# app.add_middleware(RequestBodyLoggerMiddleware)
app.add_middleware(RequestBodyLoggerMiddleware)

data_table_columns = [
    # {"label": "Status", "value": "status", "sortable": True},
    {"label": "Provider", "value": "provider", "sortable": True},
    {"label": "Base url", "value": "base_url", "sortable": True},
    # {"label": "Engine", "value": "engine", "sortable": True},
    {"label": "Tools", "value": "tools", "sortable": True},
]

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTML(
        Head(title="登录"),
        Body(
            Div(
                form.Form(
                    form.FormField("API Key", "x_api_key", type="password", placeholder="输入API密钥", required=True),
                    Div(id="error-message", class_="text-red-500 mt-2"),
                    Div(
                        button.button("提交", variant="primary", type="submit"),
                        class_="flex justify-end mt-4"
                    ),
                    hx_post="/verify-api-key",
                    hx_target="#error-message",
                    hx_swap="innerHTML",
                    class_="space-y-4"
                ),
                class_="container mx-auto p-4 max-w-md"
            )
        )
    ).render()


@app.post("/verify-api-key", response_class=HTMLResponse)
async def verify_api_key(x_api_key: str = FastapiForm(...)):
    if x_api_key == app.state.admin_api_key:  # 替换为实际的管理员API密钥
        response = JSONResponse(content={"success": True})
        response.headers["HX-Redirect"] = "/"  # 添加这一行
        response.set_cookie(
            key="x_api_key",
            value=x_api_key,
            httponly=True,
            max_age=1800,  # 30分钟
            secure=False,  # 在开发环境中设置为False，生产环境中使用HTTPS时设置为True
            samesite="lax"  # 改为"lax"以允许重定向时携带cookie
        )
        return response
    else:
        return Div("无效的API密钥", class_="text-red-500").render()

async def get_api_key(request: Request, x_api_key: Optional[str] = Depends(api_key_header)):
    if not x_api_key:
        x_api_key = request.cookies.get("x_api_key") or request.query_params.get("x_api_key")
    # print(f"Cookie x_api_key: {request.cookies.get('x_api_key')}")  # 添加此行
    # print(f"Query param x_api_key: {request.query_params.get('x_api_key')}")  # 添加此行
    # print(f"Header x_api_key: {x_api_key}")  # 添加此行
    # logger.info(f"x_api_key: {x_api_key} {x_api_key == 'your_admin_api_key'}")

    if x_api_key == app.state.admin_api_key:  # 替换为实际的管理员API密钥
        return x_api_key
    else:
        return None

@app.get("/", response_class=HTMLResponse)
async def root(x_api_key: str = Depends(get_api_key)):
    if not x_api_key:
        return RedirectResponse(url="/login", status_code=303)

    result = HTML(
        Head(
            Script("""
                document.addEventListener('DOMContentLoaded', function() {
                    const filterInput = document.getElementById('users-table-filter');
                    filterInput.addEventListener('input', function() {
                        const filterValue = this.value;
                        htmx.ajax('GET', `/filter-table?filter=${filterValue}`, '#users-table');
                    });
                });
            """),
            title="Menubar Example"
        ),
        Body(
            Div(
                Menubar(
                    MenubarMenu(
                        MenubarTrigger("File", "file-menu"),
                        MenubarContent(
                            MenubarItem("New Tab", shortcut="⌘T"),
                            MenubarItem("New Window", shortcut="⌘N"),
                            MenubarItem("New Incognito Window", disabled=True),
                            MenubarSeparator(),
                            MenubarItem("Print...", shortcut="⌘P"),
                        ),
                        id="file-menu"
                    ),
                    MenubarMenu(
                        MenubarTrigger("Edit", "edit-menu"),
                        MenubarContent(
                            MenubarItem("Undo", shortcut="⌘Z"),
                            MenubarItem("Redo", shortcut="⇧⌘Z"),
                            MenubarSeparator(),
                            MenubarItem("Cut"),
                            MenubarItem("Copy"),
                            MenubarItem("Paste"),
                        ),
                        id="edit-menu"
                    ),
                    MenubarMenu(
                        MenubarTrigger("View", "view-menu"),
                        MenubarContent(
                            MenubarItem("Always Show Bookmarks Bar"),
                            MenubarItem("Always Show Full URLs"),
                            MenubarSeparator(),
                            MenubarItem("Reload", shortcut="⌘R"),
                            MenubarItem("Force Reload", shortcut="⇧⌘R", disabled=True),
                            MenubarSeparator(),
                            MenubarItem("Toggle Fullscreen"),
                            MenubarItem("Hide Sidebar"),
                        ),
                        id="view-menu"
                    ),
                ),
                class_="p-4"
            ),
            Div(
                data_table(data_table_columns, app.state.config["providers"], "users-table"),
                class_="p-4"
            ),
            Div(id="sheet-container"),  # 这里是 sheet 将被加载的地方
            class_="container mx-auto",
            id="body"
        )
    ).render()
    # print(result)
    return result

@app.get("/dropdown-menu/{menu_id}/{row_id}", response_class=HTMLResponse)
async def get_columns_menu(menu_id: str, row_id: str):
    columns = [
        {
            "label": "Edit",
            "value": "edit",
            "hx-get": f"/edit-sheet/{row_id}",
            "hx-target": "#sheet-container",
            "hx-swap": "innerHTML"
        },
        {
            "label": "Duplicate",
            "value": "duplicate",
            "hx-post": f"/duplicate/{row_id}",
            "hx-target": "body",
            "hx-swap": "outerHTML"
        },
        {
            "label": "Delete",
            "value": "delete",
            "hx-delete": f"/delete/{row_id}",
            "hx-target": "body",
            "hx-swap": "outerHTML",
            "hx-confirm": "确定要删除这个配置吗？"
        },
    ]
    result = dropdown.dropdown_menu_content(menu_id, columns).render()
    print(result)
    return result

@app.get("/dropdown-menu/{menu_id}", response_class=HTMLResponse)
async def get_columns_menu(menu_id: str):
    result = dropdown.dropdown_menu_content(menu_id, data_table_columns).render()
    print(result)
    return result

@app.get("/filter-table", response_class=HTMLResponse)
async def filter_table(filter: str = ""):
    filtered_data = [
        provider for provider in app.state.config["providers"]
        if filter.lower() in str(provider["provider"]).lower() or
           filter.lower() in str(provider["base_url"]).lower() or
           filter.lower() in str(provider["tools"]).lower()
    ]
    return data_table(data_table_columns, filtered_data, "users-table", with_filter=False).render()

@app.post("/add-model", response_class=HTMLResponse)
async def add_model():
    new_model_id = f"model{hash(str(time.time()))}"  # 生成一个唯一的ID
    new_model = model_config_row(new_model_id).render()
    return new_model

@app.get("/edit-sheet/{row_id}", response_class=HTMLResponse)
async def get_edit_sheet(row_id: str, x_api_key: str = Depends(get_api_key)):
    row_data = get_row_data(row_id)
    print("row_data", row_data)

    model_list = []
    for index, model in enumerate(row_data["model"]):
        if isinstance(model, str):
            model_list.append(model_config_row(f"model{index}", model, "", True))
        if isinstance(model, dict):
            # print("model", model, list(model.items())[0])
            key, value = list(model.items())[0]
            model_list.append(model_config_row(f"model{index}", key, value, True))

    sheet_id = "edit-sheet"
    edit_sheet_content = sheet.SheetContent(
        sheet.SheetHeader(
            sheet.SheetTitle("Edit Item"),
            sheet.SheetDescription("Make changes to your item here.")
        ),
        sheet.SheetBody(
            Div(
                form.Form(
                    form.FormField("Provider", "provider", value=row_data["provider"], placeholder="Enter provider name", required=True),
                    form.FormField("Base URL", "base_url", value=row_data["base_url"], placeholder="Enter base URL", required=True),
                    form.FormField("API Key", "api_key", value=row_data["api"], type="text", placeholder="Enter API key"),
                    Div(
                        Div("Models", class_="text-lg font-semibold mb-2"),
                        Div(
                            *model_list,
                            id="models-container"
                        ),
                        button.button(
                            "Add Model",
                            class_="mt-2",
                            hx_post="/add-model",
                            hx_target="#models-container",
                            hx_swap="beforeend"
                        ),
                        class_="mb-4"
                    ),
                    Div(
                        checkbox.checkbox("tools", "Enable Tools", checked=row_data["tools"], name="tools"),
                        class_="mb-4"
                    ),
                    form.FormField("Notes", "notes", value=row_data.get("notes", ""), placeholder="Enter any additional notes"),
                    Div(
                        button.button("Submit", variant="primary", type="submit"),
                        button.button("Cancel", variant="outline", type="button", class_="ml-2", onclick=f"toggleSheet('{sheet_id}')"),
                        class_="flex justify-end mt-4"
                    ),
                    hx_post=f"/submit/{row_id}",
                    hx_swap="outerHTML",
                    hx_target="body",
                    class_="space-y-4"
                ),
                class_="container mx-auto p-4 max-w-2xl"
            )
        )
    )

    result = sheet.Sheet(
        sheet_id,
        Div(),
        edit_sheet_content,
        width="80%",
        max_width="800px"
    ).render()
    return result

@app.get("/add-provider-sheet", response_class=HTMLResponse)
async def get_add_provider_sheet():
    edit_sheet_content = sheet.SheetContent(
        sheet.SheetHeader(
            sheet.SheetTitle("Add New Provider"),
            sheet.SheetDescription("Enter details for the new provider.")
        ),
        sheet.SheetBody(
            Div(
                form.Form(
                    form.FormField("Provider", "provider", placeholder="Enter provider name", required=True),
                    form.FormField("Base URL", "base_url", placeholder="Enter base URL", required=True),
                    form.FormField("API Key", "api_key", type="text", placeholder="Enter API key"),
                    Div(
                        Div("Models", class_="text-lg font-semibold mb-2"),
                        Div(id="models-container"),
                        button.button(
                            "Add Model",
                            class_="mt-2",
                            hx_post="/add-model",
                            hx_target="#models-container",
                            hx_swap="beforeend"
                        ),
                        class_="mb-4"
                    ),
                    Div(
                        checkbox.checkbox("tools", "Enable Tools", name="tools"),
                        class_="mb-4"
                    ),
                    form.FormField("Notes", "notes", placeholder="Enter any additional notes"),
                    Div(
                        button.button("Submit", variant="primary", type="submit"),
                        button.button("Cancel", variant="outline", class_="ml-2"),
                        class_="flex justify-end mt-4"
                    ),
                    hx_post="/submit/new",
                    hx_swap="outerHTML",
                    hx_target="body",
                    class_="space-y-4"
                ),
                class_="container mx-auto p-4 max-w-2xl"
            )
        )
    )

    result = sheet.Sheet(
        "add-provider-sheet",
        Div(),
        edit_sheet_content,
        width="80%",
        max_width="800px"
    ).render()
    return result

def get_row_data(row_id):
    index = int(row_id)
    # print(app.state.config["providers"])
    return app.state.config["providers"][index]

def update_row_data(row_id, updated_data):
    print(row_id, updated_data)
    index = int(row_id)
    app.state.config["providers"][index] = updated_data
    with open("./api1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(app.state.config, f)

@app.post("/submit/{row_id}", response_class=HTMLResponse)
async def submit_form(
    row_id: str,
    request: Request,
    provider: str = FastapiForm(...),
    base_url: str = FastapiForm(...),
    api_key: Optional[str] = FastapiForm(None),
    tools: Optional[str] = FastapiForm(None),
    notes: Optional[str] = FastapiForm(None),
    x_api_key: str = Depends(get_api_key)
):
    form_data = await request.form()

    # 收集模型数据
    models = []
    for key, value in form_data.items():
        if key.startswith("model_name_"):
            model_id = key.split("_")[-1]
            enabled = form_data.get(f"model_enabled_{model_id}") == "on"
            rename = form_data.get(f"model_rename_{model_id}")
            if value:
                if rename:
                    models.append({value: rename})
                else:
                    models.append(value)

    updated_data = {
        "provider": provider,
        "base_url": base_url,
        "api": api_key,
        "model": models,
        "tools": tools == "on",
        "notes": notes,
    }

    print("updated_data", updated_data)

    if row_id == "new":
        # 添加新提供者
        app.state.config["providers"].append(updated_data)
    else:
        # 更新现有提供者
        update_row_data(row_id, updated_data)

    # 保存更新后的配置
    with open("./api1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(app.state.config, f)

    return await root()

@app.post("/duplicate/{row_id}", response_class=HTMLResponse)
async def duplicate_row(row_id: str):
    index = int(row_id)
    original_data = app.state.config["providers"][index]
    new_data = original_data.copy()
    new_data["provider"] += "-copy"
    app.state.config["providers"].insert(index + 1, new_data)

    # 保存更新后的配置
    with open("./api1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(app.state.config, f)

    return await root()

@app.delete("/delete/{row_id}", response_class=HTMLResponse)
async def delete_row(row_id: str):
    index = int(row_id)
    del app.state.config["providers"][index]

    # 保存更新后的配置
    with open("./api1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(app.state.config, f)

    return await root()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)
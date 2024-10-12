from fastapi import FastAPI, Form as FastAPIForm
from fastapi.responses import HTMLResponse
from xue import HTML, Head, Body, Div, xue_initialize, Strong, Span, Ul, Li
from xue.components import form, button, checkbox, input
from xue.components.model_config_row import model_config_row
from typing import List, Optional
import time

xue_initialize(tailwind=True)
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    result = HTML(
        Head(
            title="Provider Configuration Form"
        ),
        Body(
            Div(
                form.Form(
                    form.FormField("Provider", "provider", placeholder="Enter provider name", required=True),
                    form.FormField("Base URL", "base_url", placeholder="Enter base URL", required=True),
                    form.FormField("API Key", "api_key", type="password", placeholder="Enter API key"),
                    Div(
                        Div("Models", class_="text-lg font-semibold mb-2"),
                        Div(
                            model_config_row("model1", "gpt-4o: deepbricks-gpt-4o-mini", True),
                            model_config_row("model2", "gpt-4o"),
                            model_config_row("model3", "gpt-3.5-turbo"),
                            model_config_row("model4", "claude-3-5-sonnet-20240620: claude-3-5-sonnet"),
                            model_config_row("model5", "o1-mini-all"),
                            model_config_row("model6", "o1-preview-all"),
                            model_config_row("model7", "whisper-1"),
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
                        checkbox.checkbox("tools", "Enable Tools", checked=True),
                        class_="mb-4"
                    ),
                    form.FormField("Notes", "notes", placeholder="Enter any additional notes"),
                    Div(
                        button.button("Submit", class_="bg-blue-500 text-white"),
                        button.button("Cancel", class_="bg-gray-300 text-gray-700 ml-2"),
                        class_="flex justify-end mt-4"
                    ),
                    hx_post="/submit",
                    hx_swap="outerHTML",
                    class_="space-y-4"
                ),
                class_="container mx-auto p-4 max-w-2xl"
            )
        )
    ).render()
    print(result)
    return result

@app.post("/add-model", response_class=HTMLResponse)
async def add_model():
    new_model_id = f"model{hash(str(time.time()))}"  # 生成一个唯一的ID
    new_model = model_config_row(new_model_id).render()
    return new_model

def form_success_message(provider, base_url, api_key, models, tools_enabled, notes):
    return Div(
        Strong("Success!", class_="font-bold"),
        Span("Form submitted successfully.", class_="block sm:inline"),
        Ul(
            Li(f"Provider: {provider}"),
            Li(f"Base URL: {base_url}"),
            Li(f"API Key: {'*' * len(api_key)}"),
            Li(f"Models: {', '.join(models)}"),
            Li(f"Tools Enabled: {'Yes' if tools_enabled else 'No'}"),
            Li(f"Notes: {notes}"),
            class_="mt-3"
        ),
        class_="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative",
        role="alert"
    )

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    provider: str = FastAPIForm(...),
    base_url: str = FastAPIForm(...),
    api_key: str = FastAPIForm(...),
    models: List[str] = FastAPIForm([]),
    tools: Optional[str] = FastAPIForm(None),
    notes: Optional[str] = FastAPIForm(None)
):
    # 处理提交的数据
    print(f"Received: provider={provider}, base_url={base_url}, api_key={api_key}")
    print(f"Models: {models}")
    print(f"Tools Enabled: {tools is not None}")
    print(f"Notes: {notes}")

    # 返回处理结果
    return form_success_message(
        provider,
        base_url,
        api_key,
        models,
        tools is not None,
        notes or "No notes provided"
    ).render()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)
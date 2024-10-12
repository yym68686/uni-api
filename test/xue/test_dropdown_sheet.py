from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from xue import HTML, Head, Body, Div, xue_initialize, Script
from xue.components import dropdown, sheet, button, form, input

xue_initialize(tailwind=True)
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    result = HTML(
        Head(
            title="Dropdown with Edit Sheet Example",
        ),
        Body(
            Div(
                dropdown.dropdown_menu("Actions"),
                Div(id="sheet-container"),  # 这里是 sheet 将被加载的地方
                class_="container mx-auto p-4"
            )
        )
    ).render()
    print(result)
    return result

@app.get("/dropdown-menu/{menu_id}", response_class=HTMLResponse)
async def get_dropdown_menu_content(menu_id: str):
    items = [
        {
            "icon": "pencil",
            "label": "Edit",
            "hx-get": "/edit-sheet",
            "hx-target": "#sheet-container",
            "hx-swap": "innerHTML"
        },
        {"icon": "trash", "label": "Delete"},
        {"icon": "copy", "label": "Duplicate"},
    ]
    result = dropdown.dropdown_menu_content(menu_id, items).render()
    print("dropdown-menu result", result)
    return result

@app.get("/edit-sheet", response_class=HTMLResponse)
async def get_edit_sheet():
    edit_sheet_content = sheet.SheetContent(
        sheet.SheetHeader(
            sheet.SheetTitle("Edit Item"),
            sheet.SheetDescription("Make changes to your item here.")
        ),
        sheet.SheetBody(
            form.Form(
                form.FormField("Name", "name", placeholder="Enter item name"),
                form.FormField("Description", "description", placeholder="Enter item description"),
                Div(
                    button.button("Save", class_="bg-blue-500 text-white"),
                    button.button("Cancel", class_="bg-gray-300 text-gray-700 ml-2", data_close_sheet="true"),
                    class_="flex justify-end mt-4"
                ),
                class_="space-y-4"
            )
        )
    )

    result = sheet.Sheet(
        "edit-sheet",
        Div(),
        edit_sheet_content,
        width="80%",
        max_width="800px"
    ).render()
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)
from xue import Div, Table, Thead, Tbody, Tr, Th, Td, Button, Input, Script, Head, Style, Span
from xue.components.checkbox import checkbox
from xue.components.dropdown import dropdown_menu, dropdown_menu_content
from xue.components.button import button
from xue.components.input import input

Head.add_default_children([
    Style("""
        .data-table-container {
            width: 100%;
            overflow-x: auto;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            overflow-x: visible !important;
        }
        .data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }
        .data-table th, .data-table td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        .data-table th {
            font-weight: 500;
            font-size: 0.875rem;
            color: #4b5563;
            height: 2.5rem;
            transition: background-color 0.2s;
        }
        .data-table thead tr:hover th,
        .data-table tbody tr:hover {
            background-color: #f8fafc;
        }
        .data-table tbody tr:last-child td {
            border-bottom: none;
        }
        .sortable-header {
            cursor: pointer;
            user-select: none;
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            transition: background-color 0.2s;
        }
        .sortable-header:hover {
            background-color: #e5e7eb;
        }
        .sort-icon {
            display: inline-block;
            width: 1rem;
            height: 1rem;
            margin-left: 0.25rem;
            transition: transform 0.2s;
            opacity: 0;
        }
        .sortable-header:hover .sort-icon,
        .sort-asc .sort-icon,
        .sort-desc .sort-icon {
            opacity: 1;
        }
        .sort-asc .sort-icon {
            transform: rotate(180deg);
        }
        .table-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .table-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1rem;
        }
        .pagination {
            display: flex;
            gap: 0.5rem;
        }
        @media (prefers-color-scheme: dark) {
            .data-table-container {
                border-color: #4b5563;
            }
            .data-table th, .data-table td {
                border-color: #4b5563;
            }
            .data-table th {
                color: #d1d5db;
            }
            .data-table thead tr:hover th,
            .data-table tbody tr:hover {
                background-color: #1f2937;
            }
            .sortable-header:hover {
                background-color: #374151;
            }
        }
    """, id="data-table-style"),
    Script("""
        function toggleAllRows(checked) {
            const checkboxes = document.querySelectorAll('.row-checkbox');
            checkboxes.forEach(cb => cb.checked = checked);
            updateSelectedCount();
        }

        function updateSelectedCount() {
            const selectedCount = document.querySelectorAll('.row-checkbox:checked').length;
            const totalCount = document.querySelectorAll('.row-checkbox').length;
            document.getElementById('selected-count').textContent = `${selectedCount} of ${totalCount} row(s) selected.`;
        }

        function sortTable(columnIndex, accessor) {
            const table = document.querySelector('.data-table');
            const header = table.querySelector(`th[data-accessor="${accessor}"]`);
            const isAscending = !header.classList.contains('sort-asc');

            // Update sort direction
            table.querySelectorAll('th').forEach(th => th.classList.remove('sort-asc', 'sort-desc'));
            header.classList.add(isAscending ? 'sort-asc' : 'sort-desc');

            // Sort the table
            const rows = Array.from(table.querySelectorAll('tbody tr'));
            rows.sort((a, b) => {
                const aValue = a.querySelector(`td[data-accessor="${accessor}"]`).textContent;
                const bValue = b.querySelector(`td[data-accessor="${accessor}"]`).textContent;
                return isAscending ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
            });

            // Update the table
            const tbody = table.querySelector('tbody');
            rows.forEach(row => tbody.appendChild(row));
        }

        document.addEventListener('change', function(event) {
            if (event.target.classList.contains('row-checkbox')) {
                updateSelectedCount();
            }
        });
    """, id="data-table-script"),
])

def data_table(columns, data, id, with_filter=True):
    return Div(
        Div(
            input(type="text", placeholder="Filter...", id=f"{id}-filter", class_="mr-auto"),
            Div(
                button(
                    "Add Provider",
                    variant="secondary",
                    hx_get="/add-provider-sheet",
                    hx_target="#sheet-container",
                    hx_swap="innerHTML",
                    class_="h-[2.625rem]"
                ),
                dropdown_menu("Columns"),
            ),
            class_="table-header flex items-center"
        ) if with_filter else None,
        Div(
            Div(
                Table(
                    Thead(
                        Tr(
                            Th(checkbox("select-all", "", onclick="toggleAllRows(this.checked)")),
                            *[Th(
                                Div(
                                    col['label'],
                                    Span("▼", class_="sort-icon"),
                                    class_="sortable-header" if col.get('sortable', False) else "",
                                    onclick=f"sortTable({i}, '{col['value']}')" if col.get('sortable', False) else None
                                ),
                                data_accessor=col['value']
                            ) for i, col in enumerate(columns)],
                            Th("Actions")  # 新增的操作列
                        )
                    ),
                    Tbody(
                        *[Tr(
                            Td(checkbox(f"row-{i}", "", class_="row-checkbox")),
                            *[Td(row[col['value']], data_accessor=col['value']) for col in columns],
                            Td(row_actions_menu(i)),  # 使用行索引作为 row_id
                            id=f"row-{i}"
                        ) for i, row in enumerate(data)]
                    ),
                    class_="data-table"
                ),
                class_="data-table-container"
            ),
            Div(
                Div(id="selected-count", class_="text-sm text-gray-500"),
                Div(
                    button("Previous", variant="outline", class_="mr-2"),
                    button("Next", variant="outline"),
                    class_="pagination"
                ),
                class_="table-footer"
            ),
            id=id
        ),
    )

def get_column_visibility_menu(id, columns):
    return dropdown_menu_content(id, [
        {"label": col['label'], "value": col['value']}
        for col in columns if col.get('can_hide', True)
    ])

def row_actions_menu(row_id):
    return dropdown_menu("⋮", id=f"row-actions-menu-{row_id}", hx_get=f"/dropdown-menu/dropdown-menu-⋮/{row_id}")

def get_row_actions_menu(row_id):
    return dropdown_menu_content(f"row-actions-{row_id}", [
        {"label": "Edit", "icon": "pencil"},
        {"label": "Duplicate", "icon": "copy"},
        {"label": "Delete", "icon": "trash"},
        "separator",
        {"label": "More...", "icon": "more-horizontal"},
    ])

def render_row(row_data, row_id, columns):
    return Tr(
        Td(checkbox(f"row-{row_id}", "", class_="row-checkbox")),
        *[Td(row_data[col['value']], data_accessor=col['value']) for col in columns],
        Td(row_actions_menu(row_id)),
        id=f"row-{row_id}"
    ).render()
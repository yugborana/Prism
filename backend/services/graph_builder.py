from typing import Any, Dict, Tuple

import networkx as nx


def build_simple_graph(tree, source_code: str, lang: str, file_path: str) -> nx.DiGraph:
    """
    Build a simple semantic graph with just nodes and basic relationships.
    """
    graph = nx.DiGraph()

    def node_text(node):
        try:
            return source_code[node.start_byte : node.end_byte]
        except Exception:
            return ""

    def walk(node, current_function=None):
        # Track functions and classes
        if lang == "python":
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    func_label = f"{file_path}::function::{name}"
                    graph.add_node(func_label, type="function", name=name)
                    if current_function:
                        graph.add_edge(current_function, func_label, type="contains")
                    current_function = func_label

            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    class_label = f"{file_path}::class::{name}"
                    graph.add_node(class_label, type="class", name=name)
                    if current_function:
                        graph.add_edge(current_function, class_label, type="contains")

        elif lang == "go":
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    func_label = f"{file_path}::function::{name}"
                    graph.add_node(func_label, type="function", name=name)
                    if current_function:
                        graph.add_edge(current_function, func_label, type="contains")
                    current_function = func_label

            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    func_label = f"{file_path}::method::{name}"
                    graph.add_node(func_label, type="method", name=name)
                    if current_function:
                        graph.add_edge(current_function, func_label, type="contains")
                    current_function = func_label

            elif node.type == "type_spec":
                name_node = node.child_by_field_name("name")
                type_node = node.child_by_field_name("type")
                if name_node and type_node:
                    name = node_text(name_node)
                    type_kind = type_node.type
                    if type_kind in ("struct_type", "interface_type"):
                        class_label = f"{file_path}::type::{name}"
                        graph.add_node(class_label, type=type_kind, name=name)
                        if current_function:
                            graph.add_edge(current_function, class_label, type="contains")

        elif lang == "rust":
            if node.type == "function_item":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    func_label = f"{file_path}::function::{name}"
                    graph.add_node(func_label, type="function", name=name)
                    if current_function:
                        graph.add_edge(current_function, func_label, type="contains")
                    current_function = func_label

            elif node.type == "struct_item":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    class_label = f"{file_path}::struct::{name}"
                    graph.add_node(class_label, type="struct", name=name)
                    if current_function:
                        graph.add_edge(current_function, class_label, type="contains")

            elif node.type == "impl_item":
                type_node = node.child_by_field_name("type")
                if type_node:
                    name = node_text(type_node)
                    class_label = f"{file_path}::impl::{name}"
                    graph.add_node(class_label, type="impl", name=name)
                    if current_function:
                        graph.add_edge(current_function, class_label, type="contains")

            elif node.type == "trait_item":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    class_label = f"{file_path}::trait::{name}"
                    graph.add_node(class_label, type="trait", name=name)
                    if current_function:
                        graph.add_edge(current_function, class_label, type="contains")

        elif lang in ("javascript", "typescript"):
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    func_label = f"{file_path}::function::{name}"
                    graph.add_node(func_label, type="function", name=name)
                    if current_function:
                        graph.add_edge(current_function, func_label, type="contains")
                    current_function = func_label

            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    class_label = f"{file_path}::class::{name}"
                    graph.add_node(class_label, type="class", name=name)
                    if current_function:
                        graph.add_edge(current_function, class_label, type="contains")

            elif node.type == "method_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    func_label = f"{file_path}::method::{name}"
                    graph.add_node(func_label, type="method", name=name)
                    if current_function:
                        graph.add_edge(current_function, func_label, type="contains")
                    current_function = func_label

            elif node.type == "arrow_function":
                parent = node.parent
                if parent and parent.type == "variable_declarator":
                    name_node = parent.child_by_field_name("name")
                    if name_node:
                        name = node_text(name_node)
                        func_label = f"{file_path}::function::{name}"
                        graph.add_node(func_label, type="arrow_function", name=name)
                        if current_function:
                            graph.add_edge(current_function, func_label, type="contains")
                        current_function = func_label

            elif node.type == "function_expression":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    func_label = f"{file_path}::function::{name}"
                    graph.add_node(func_label, type="function", name=name)
                    if current_function:
                        graph.add_edge(current_function, func_label, type="contains")
                    current_function = func_label

        # Track imports
        if lang == "python" and node.type in (
            "import_statement",
            "import_from_statement",
        ):
            import_text = node_text(node).strip()
            imp_label = f"{file_path}::import::{import_text}"
            graph.add_node(imp_label, type="import", text=import_text)

            # Connect import to file or function
            if current_function:
                graph.add_edge(current_function, imp_label, type="uses")
            else:
                file_anchor = f"{file_path}::file"
                if file_anchor not in graph:
                    graph.add_node(file_anchor, type="file")
                graph.add_edge(file_anchor, imp_label, type="imports")

        elif lang == "go" and node.type == "import_spec":
            import_text = node_text(node).strip()
            imp_label = f"{file_path}::import::{import_text}"
            graph.add_node(imp_label, type="import", text=import_text)

            # Connect import to file or function
            if current_function:
                graph.add_edge(current_function, imp_label, type="uses")
            else:
                file_anchor = f"{file_path}::file"
                if file_anchor not in graph:
                    graph.add_node(file_anchor, type="file")
                graph.add_edge(file_anchor, imp_label, type="imports")

        elif lang == "rust" and node.type == "use_declaration":
            import_text = node_text(node).strip()
            imp_label = f"{file_path}::import::{import_text}"
            graph.add_node(imp_label, type="import", text=import_text)

            # Connect import to file or function
            if current_function:
                graph.add_edge(current_function, imp_label, type="uses")
            else:
                file_anchor = f"{file_path}::file"
                if file_anchor not in graph:
                    graph.add_node(file_anchor, type="file")
                graph.add_edge(file_anchor, imp_label, type="imports")

        elif lang in ("javascript", "typescript") and node.type == "import_statement":
            import_text = node_text(node).strip()
            imp_label = f"{file_path}::import::{import_text}"
            graph.add_node(imp_label, type="import", text=import_text)

            # Connect imports to file or function
            if current_function:
                graph.add_edge(current_function, imp_label, type="uses")
            else:
                file_anchor = f"{file_path}::file"
                if file_anchor not in graph:
                    graph.add_node(file_anchor, type="file")
                graph.add_edge(file_anchor, imp_label, type="imports")

        # Track function calls (basic)
        if lang == "python" and node.type == "call":
            func_node = None
            if node.child_by_field_name("function"):
                func_node = node.child_by_field_name("function")
            elif node.children:
                func_node = node.children[0]

            if func_node and current_function:
                called_name = node_text(func_node).split("(")[0].strip()
                if called_name and called_name.replace("_", "").isalnum():
                    call_label = f"{file_path}::call::{called_name}"
                    graph.add_node(call_label, type="call", name=called_name)
                    graph.add_edge(current_function, call_label, type="calls")

        elif lang == "go" and node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node and current_function:
                called_name = node_text(func_node).strip()
                if called_name:
                    call_label = f"{file_path}::call::{called_name}"
                    graph.add_node(call_label, type="call", name=called_name)
                    graph.add_edge(current_function, call_label, type="calls")

        elif lang == "rust" and node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node and current_function:
                called_name = node_text(func_node).strip()
                if called_name:
                    call_label = f"{file_path}::call::{called_name}"
                    graph.add_node(call_label, type="call", name=called_name)
                    graph.add_edge(current_function, call_label, type="calls")

        elif lang in ("javascript", "typescript") and node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node and current_function:
                called_name = node_text(func_node).strip()
                if called_name:
                    call_label = f"{file_path}::call::{called_name}"
                    graph.add_node(call_label, type="call", name=called_name)
                    graph.add_edge(current_function, call_label, type="calls")

        for child in node.children:
            walk(child, current_function)

    walk(tree.root_node)
    return graph


# def analyze_cross_file_imports(
#     parsed_files: Dict[str, Tuple], graph: nx.DiGraph
# ) -> Dict[str, Any]:
#     import_edges = []
#     file_imports = {}
#
#     for file_path, (tree, source_code) in parsed_files.items():
#         imports = []
#
#         # Extract imports from the file
#         def node_text(node):
#             try:
#                 return source_code[node.start_byte : node.end_byte]
#             except Exception:
#                 return ""
#
#         def extract_imports(node):
#             if node.type == "import_from_statement":
#                 for child in node.children:
#                     if child.type == "dotted_name":
#                         module = node_text(child)
#                         imports.append(module)
#                         break
#             elif node.type == "import_statement":
#                 for child in node.children:
#                     if child.type == "dotted_name":
#                         module = node_text(child)
#                         imports.append(module)
#
#             for child in node.children:
#                 extract_imports(child)
#
#         extract_imports(tree.root_node)
#         file_imports[file_path] = imports
#
#         # Add import edges to graph
#         for import_name in imports:
#             # Simple heuristic: if import looks like a file, add edge
#             if "." in import_name:
#                 target_file = f"src/{import_name.replace('.', '/')}.py"
#                 import_edges.append(
#                     {
#                         "source": file_path,
#                         "target": target_file,
#                         "import_name": import_name,
#                         "type": "import",
#                     }
#                 )
#
#     return {
#         "file_imports": file_imports,
#         "import_edges": import_edges,
#         "total_imports": sum(len(imports) for imports in file_imports.values()),
#         "graph_stats": {
#             "nodes": graph.number_of_nodes(),
#             "edges": graph.number_of_edges(),
#         },
#     }

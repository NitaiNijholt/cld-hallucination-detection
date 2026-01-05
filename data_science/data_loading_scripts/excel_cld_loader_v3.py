import os
import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
from pathlib import Path
import tempfile

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def load_cld_from_excel(excel_file, plot_graph=False, export_json=False, output_dir=None):
    """
    Load a Causal Loop Diagram from an Excel file and optionally export to JSON.
    
    Args:
        excel_file: Path to the Excel file
        plot_graph: Whether to visualize the graph
        export_json: Whether to export the CLD data to JSON files
        output_dir: Directory to save JSON files (if None, uses same directory as Excel file)
        
    Returns:
        dict: Dictionary containing graph, variables DataFrame, edges DataFrame, 
              variable definitions dictionary, and relationships dictionary
    """
    print(f"Loading CLD from Excel file: {excel_file}")
    
    # Check if file exists
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Excel file not found: {excel_file}")
    
    # Set output directory if not specified
    if export_json and output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(excel_file))
    elif export_json:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the Excel file
    try:
        xls = pd.ExcelFile(excel_file)
        
        # Check for required sheets
        required_sheets = ["Variable_definitions", "Variable_links"]
        missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]
        
        if missing_sheets:
            raise ValueError(f"Excel file is missing required sheets: {', '.join(missing_sheets)}")
        
        # Load variables
        var_df = pd.read_excel(excel_file, sheet_name="Variable_definitions")
        print(f"Loaded {len(var_df)} variables")
        
        # Create a dictionary for easy access to variable definitions
        variable_definitions = {}
        for _, row in var_df.iterrows():
            variable = row["Variable"]
            definition = row["Definition"] if "Definition" in var_df.columns else ""
            variable_definitions[variable] = definition
        
        # Load edges
        edge_df = pd.read_excel(excel_file, sheet_name="Variable_links")
        print(f"Loaded {len(edge_df)} edges")
        
        # Create a dictionary for easy access to relationship types
        relationship_types = {}
        for _, row in edge_df.iterrows():
            source = row["Source"]
            target = row["Target"]
            
            # Check if the polarity column exists
            polarity = "unknown"
            if "Polarity" in edge_df.columns and pd.notna(row["Polarity"]):
                polarity = row["Polarity"].lower()
                
            # Store the relationship info
            relationship_types[(source, target)] = {
                "polarity": polarity,
                "relationship_type": map_polarity_to_relationship(polarity)
            }
        
        # Check if context sheet exists
        context_data = {}
        if "Context" in xls.sheet_names:
            context_df = pd.read_excel(excel_file, sheet_name="Context", header=None)
            for _, row in context_df.iterrows():
                if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                    context_data[row.iloc[0]] = row.iloc[1]
            print(f"Loaded context data: {context_data}")
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for var in variable_definitions:
            G.add_node(var, definition=variable_definitions[var])
        
        # Add edges
        for (source, target), rel_data in relationship_types.items():
            G.add_edge(source, target, 
                      polarity=rel_data["polarity"],
                      relationship=rel_data["relationship_type"])
        
        # Export to JSON if requested
        json_files = {}
        if export_json:
            # Get base filename without extension
            base_filename = os.path.splitext(os.path.basename(excel_file))[0]
            
            # Create variable data with index key
            variables_json = []
            for i, (var, definition) in enumerate(variable_definitions.items()):
                variables_json.append({
                    "index": i,
                    "name": var,
                    "definition": definition
                })
            
            # Create edge data with index key
            edges_json = []
            for i, ((source, target), rel_data) in enumerate(relationship_types.items()):
                edges_json.append({
                    "index": i,
                    "source": source,
                    "target": target,
                    "relationship": rel_data["relationship_type"],
                    "polarity": rel_data["polarity"]
                })
            
            # Save to JSON files
            vars_json_path = os.path.join(output_dir, f"{base_filename}_vars_data.json")
            with open(vars_json_path, 'w') as f:
                json.dump(variables_json, f, indent=2)
            print(f"Variables data saved to: {vars_json_path}")
            
            edges_json_path = os.path.join(output_dir, f"{base_filename}_edges_data.json")
            with open(edges_json_path, 'w') as f:
                json.dump(edges_json, f, indent=2)
            print(f"Edges data saved to: {edges_json_path}")
            
            # Save context data if available
            if context_data:
                context_json_path = os.path.join(output_dir, f"{base_filename}_context_data.json")
                with open(context_json_path, 'w') as f:
                    json.dump(context_data, f, indent=2)
                print(f"Context data saved to: {context_json_path}")
                json_files["context"] = context_json_path
            
            json_files["variables"] = vars_json_path
            json_files["edges"] = edges_json_path
        
        # Plot the graph if requested
        if plot_graph:
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue", alpha=0.8)
            
            # Draw edges with colors based on relationship type
            edge_colors = []
            for u, v, data in G.edges(data=True):
                if data.get("relationship") == "POSITIVE":
                    edge_colors.append("green")
                elif data.get("relationship") == "NEGATIVE":
                    edge_colors.append("red")
                else:
                    edge_colors.append("gray")
            
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color=edge_colors, 
                                  arrowsize=20, connectionstyle='arc3,rad=0.1')
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
            
            plt.title("Causal Loop Diagram")
            plt.axis("off")
            plt.tight_layout()
            
            # Save the plot to a file
            output_file = os.path.splitext(excel_file)[0] + "_graph.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Graph saved to: {output_file}")
            plt.close()
        
        # Return the data with additional dictionaries for easy access
        return {
            "graph": G,
            "variables": var_df,
            "edges": edge_df,
            "variable_definitions": variable_definitions,  # Dictionary of variable:definition
            "relationship_types": relationship_types,      # Dictionary of (source,target):{polarity, type}
            "context": context_data,
            "json_files": json_files if export_json else {}
        }
    
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        raise

def map_polarity_to_relationship(polarity):
    """Map polarity string to relationship type used in the application"""
    polarity = polarity.lower() if isinstance(polarity, str) else "unknown"
    
    if polarity == "positive":
        return "POSITIVE"
    elif polarity == "negative":
        return "NEGATIVE"
    elif polarity == "delayed":
        return "DELAYED"
    elif polarity == "nonlinear":
        return "NONLINEAR"
    else:
        return "UNKNOWN"

def batch_convert_excel_to_json(excel_folder_path, output_folder_path=None, plot_graphs=False):
    """
    Convert all Excel CLD files in a folder to JSON format.
    
    Args:
        excel_folder_path: Path to folder containing Excel CLD files
        output_folder_path: Path to save JSON files (if None, uses same folder)
        plot_graphs: Whether to create graph visualizations
        
    Returns:
        dict: Dictionary mapping Excel filenames to their JSON output paths
    """
    if output_folder_path is None:
        output_folder_path = excel_folder_path
    else:
        os.makedirs(output_folder_path, exist_ok=True)
    
    results = {}
    excel_files = [f for f in os.listdir(excel_folder_path) if f.endswith('.xlsx')]
    
    for excel_file in excel_files:
        excel_path = os.path.join(excel_folder_path, excel_file)
        try:
            print(f"\nProcessing {excel_file}...")
            result = load_cld_from_excel(
                excel_path, 
                plot_graph=plot_graphs,
                export_json=True,
                output_dir=output_folder_path
            )
            
            results[excel_file] = {
                "variables_json": result["json_files"]["variables"],
                "edges_json": result["json_files"]["edges"],
                "node_count": len(result["graph"].nodes),
                "edge_count": len(result["graph"].edges)
            }
            print(f"Successfully processed {excel_file}")
        except Exception as e:
            print(f"Error processing {excel_file}: {e}")
            results[excel_file] = {"error": str(e)}
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Load a Causal Loop Diagram from Excel")
    parser.add_argument("excel_file", help="Path to the Excel file")
    parser.add_argument("--plot", action="store_true", help="Plot the graph")
    parser.add_argument("--json", action="store_true", help="Export to JSON")
    parser.add_argument("--output", help="Output directory for JSON files")
    parser.add_argument("--batch", action="store_true", help="Process all Excel files in directory")
    args = parser.parse_args()
    
    if args.batch:
        # Process all Excel files in the directory
        excel_dir = os.path.dirname(os.path.abspath(args.excel_file))
        results = batch_convert_excel_to_json(
            excel_dir,
            args.output,
            args.plot
        )
        
        # Print summary
        print("\nBatch Processing Summary:")
        for file, data in results.items():
            if "error" in data:
                print(f"  {file}: ERROR - {data['error']}")
            else:
                print(f"  {file}: {data['node_count']} nodes, {data['edge_count']} edges")
    else:
        # Load a single CLD from Excel
        cld_data = load_cld_from_excel(
            args.excel_file, 
            plot_graph=args.plot,
            export_json=args.json,
            output_dir=args.output
        )
        
        # Print summary
        print("\nCLD Summary:")
        print(f"Number of variables: {len(cld_data['variables'])}")
        print(f"Number of edges: {len(cld_data['edges'])}")
        
        if cld_data["context"]:
            print("\nContext:")
            for key, value in cld_data["context"].items():
                print(f"  {key}: {value}")
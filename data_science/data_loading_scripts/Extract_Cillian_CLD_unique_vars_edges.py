import json

# Define the data as a list of tuples
data = [
    ("Depression", "Sleep"),
    ("Depression", "Heart_Disease"),
    ("Frailty", "Stairs"),
    ("Frailty", "Pain"),
    ("Frailty", "Chair_SitRise"),
    ("Frailty", "Physical_Activity"),
    ("Frailty", "Toenail_Cut"),
    ("Frailty", "Walk_5min"),
    ("Frailty", "Undress"),
    ("Frailty", "Osteoarthritis"),
    ("Frailty", "Transport"),
    ("Frailty", "Memory_Complaints"),
    ("Frailty", "Heart_Disease"),
    ("Frailty", "Grip"),
    ("Frailty", "Arterial_Disease"),
    ("Frailty", "Respiratory"),
    ("Frailty", "Alcohol_Use"),
    ("Frailty", "CVA"),
    ("Frailty", "Diabetes"),
    ("Frailty", "Cancer"),
    ("Frailty", "Hearing"),
    ("Frailty", "Health_vs_Peers"),
    ("General_Health", "Health_vs_Peers"),
    ("General_Health", "Rheumatoid_Arthritis"),
    ("BMI", "MAP"),
    ("WHR", "Grip"),
    ("WHR", "Diabetes"),
    ("Alcohol_Use", "WHR"),
    ("Osteoarthritis", "Rheumatoid_Arthritis"),
    ("Stairs", "Chair_SitRise"),
    ("Undress", "Toenail_Cut"),
    ("Chair_SitRise", "Undress"),
    ("Chair_SitRise", "Transport"),
    ("Pain", "Osteoarthritis"),
    ("Memory_Complaints", "Hearing"),
    ("Physical_Activity", "Memory_Complaints"),
    ("Walk_5min", "Physical_Activity"),
    ("Chair_SitRise", "Pain"),
    ("Frailty", "General_Health"),
    ("Pain", "General_Health"),
    ("WHR", "BMI"),
    ("Anxiety", "Depression"),
    ("Depression", "Frailty"),
    ("Walk_5min", "Stairs"),
    ("Stairs", "BMI"),
    ("Toenail_Cut", "WHR"),
    ("Memory_Complaints", "Arterial_Disease")
]

# Create a set to store unique variables
unique_variables = set()

# Create a list of edge objects
validation_edges_list = []
for var1, var2 in data:
    unique_variables.add(var1)
    unique_variables.add(var2)
    # By default, set relationship to "CAUSES"
    edge_obj = {
        "source": var1,
        "target": var2,
        "relationship": "ASSOCIATED"
    }
    validation_edges_list.append(edge_obj)

# Sort the unique variables
unique_variables_list = sorted(unique_variables)

# Write edges to a JSON file
# NOTE: The key is "validation_edges" instead of "edges"
edges_json_data = {
    "validation_edges": validation_edges_list
}
with open("Cillian_CLD_edges_data.json", "w") as edges_file:
    json.dump(edges_json_data, edges_file, indent=4)
print("Edges successfully saved to Cillian_CLD_edges_data.json")

# Write unique variables to another JSON file
vars_json_data = {
    "unique_variables": unique_variables_list
}
with open("Cillian_CLD_unique_vars_data.json", "w") as vars_file:
    json.dump(vars_json_data, vars_file, indent=4)
print("Unique variables successfully saved to Cillian_CLD_unique_vars_data.json")

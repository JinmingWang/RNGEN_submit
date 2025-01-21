import osmnx as ox

# Configure osmnx to use caching
ox.config(use_cache=True)

# List of cities
cities = ["Shanghai, China", "Tokyo, Japan", "Las Vegas, Nevada, USA"]

# Loop through each city, retrieve the street network, and save it as a GraphML file
for city in cities:
    # Retrieve the street network for the city
    graph = ox.graph_from_place(city, network_type='drive')

    # Define a filename for saving the GraphML file
    filename = f"{city.split(',')[0].replace(' ', '_')}_street_network.graphml"

    # Save the graph as a GraphML file
    ox.save_graphml(graph, filepath=filename)

    print(f"Saved road network for {city} as {filename}")
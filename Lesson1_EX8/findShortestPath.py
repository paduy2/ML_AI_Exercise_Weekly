import os
import json
import time
from collections import deque, defaultdict
from tqdm import tqdm
from datetime import datetime
graph = None  # Global variable to store the graph

def load_words(file_path):
    """Load words from the specified file and filter out words with less than 3 letters."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    words = set()
    with open(file_path, 'r') as file:
        for line in file:
            for word in line.split():
                if len(word) >= 3:
                    words.add(word)
    return words

def build_graph(words):
    """Build a graph where each word is a node and edges exist if the suffix of one word matches the prefix of another."""
    graph = defaultdict(set)
    words_list = list(words)
    for word in tqdm(words_list, desc="Building Graph"):
        suffix = word[-2:]
        for other in words:
            if word != other and suffix == other[:2]:
                graph[word].add(other)
    return graph

def save_graph(graph, file_path):
    """Save the graph to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump({key: list(value) for key, value in graph.items()}, file)

def load_graph(file_path):
    """Load the graph from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {key: set(value) for key, value in data.items()}

def find_all_shortest_chains(start_word, end_word, graph):
    """Find all shortest chains from start_word to end_word using BFS."""
    if start_word not in graph or end_word not in graph:
        return None
    
    queue = deque([(start_word, [start_word])])
    visited = set()
    shortest_paths = []
    shortest_length = float('inf')

    while queue:
        current_word, path = queue.popleft()

        if len(path) > shortest_length:
            break
        
        if current_word == end_word:
            if not shortest_paths or len(path) == shortest_length:
                shortest_paths.append(path)
                shortest_length = len(path)
            continue

        if current_word not in visited:
            visited.add(current_word)
            for neighbor in graph.get(current_word, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return shortest_paths if shortest_paths else None

def shortest_word_chain(file_path, graph_path, word1, word2):
    """Load words, build graph, save the graph, and find the shortest chains from word1 to word2."""
    global graph
    if graph is None:
      if not os.path.exists(graph_path):
          start_time = time.time()
          words = load_words(file_path)
          graph = build_graph(words)
          save_graph(graph, graph_path)
          end_time = time.time()
          print(f"Time taken to build the graph: {end_time - start_time} seconds")
      else:
          current_datetime = datetime.now()
          print("Current date and time:", current_datetime)

          graph = load_graph(graph_path)

          current_datetime = datetime.now()
          print("Current date and time:", current_datetime)
    return find_all_shortest_chains(word1, word2, graph)

def main():
    print("Welcome to the Shortest Word Chain Finder!")
    while True:
        print("\nMenu:")
        print("1. Find the shortest word chains")
        print("2. Exit")
        choice = input("Please enter your choice (1 or 2): ")

        if choice == '1':
            word1 = input("Enter the first word: ").strip()
            word2 = input("Enter the second word: ").strip()
            try:
                results = shortest_word_chain('wordsEn.txt', 'graph.json', word1, word2)
                print("RESULT NUMBER OF PATH:", len(results))
                if results:
                    for path in results:
                        print(" -> ".join(path))
                else:
                    print("No chain found.")
            except FileNotFoundError as e:
                print(e)
        elif choice == '2':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")
        current_datetime = datetime.now()
        print("Current date and time after process main func:", current_datetime)

if __name__ == "__main__":
    main()

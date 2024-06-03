# Ex4: Read the entire file story.txt and write a program to print out top 100 words occur most
# frequently and their corresponding appearance. You could ignore all
# punction characters such as comma, dot, semicolon, ...
# Sample output:
# house: 453
# dog: 440
# people: 312
# ...
import string

def read_file(file_path):
    # """Reads the content of the file and returns it as a single string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def clean_text(text):
    # """Removes punctuation from the text and converts it to lowercase."""
    # Remove punctuation using str.translate
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).lower()

def count_word_frequencies(text):
    # """Counts the frequency of each word in the text."""
    words = text.split()
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

def get_top_words(word_count, top_n=100):
    # """Sorts the word count dictionary and returns the top `top_n` words and their frequencies."""
    sorted_word_count = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
    return sorted_word_count[:top_n]

def print_top_words(word_frequencies):
    # """Prints the words and their frequencies."""
    for word, frequency in word_frequencies:
        print(f"{word}: {frequency}")

def main():
    file_path = './story.txt'  # Path to the input file
    text = read_file(file_path)
    cleaned_text = clean_text(text)
    word_frequencies = count_word_frequencies(cleaned_text)
    top_words = get_top_words(word_frequencies)
    print_top_words(top_words)

if __name__ == "__main__":
    main()


from query import Query
from vectordb import Vectordb


def main():
    collection = Vectordb(True)  # Regenerate the collection each time for testing 
    Query(collection)

if __name__ == "__main__":
    main()
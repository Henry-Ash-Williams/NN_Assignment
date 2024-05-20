import os 
from pybtex.database import parse_file, Entry
from pybtex.utils import OrderedCaseInsensitiveDict
from IPython.display import display, Markdown 
from rich import inspect 

BIBLIOGRAPHY_PATH = os.path.join(os.getcwd(), 'references.bib')

def read_citations(bibliography_path: str) -> OrderedCaseInsensitiveDict: 
    if not os.path.exists(bibliography_path):
        raise FileNotFoundError(f"Could not find bibliography file at the path {bibliography_path}")
    bibliography = parse_file(bibliography_path)
    return bibliography.entries 

def format_citation(entry):
    fields = entry.fields
    authors = " and ".join(str(person) for person in entry.persons['author'])
    title = fields['title']
    journal = fields.get('journal', '')
    year = fields['year']
    citation = f"{authors} ({year}). *{title}*. {journal}."
    return citation

if __name__ == "__main__":
    print(format_citation(read_citations(BIBLIOGRAPHY_PATH)['dummy']))

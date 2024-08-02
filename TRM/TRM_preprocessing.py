import re
import os
import pdfplumber
import pandas as pd
import numpy as np
import numexpr as ne
import time

filepath = "Exercises/TRM/TRM.pdf"
pdf = pdfplumber.open(filepath)
pages = pdf.pages[0:479]

table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_x_tolerance": 15,
        "snap_y_tolerance": 6,
        "intersection_tolerance": 15
    }

# Generate TRM text file if not there
if not os.path.isfile("Exercises/TRM/TRM.txt"):
    with open("Exercises/TRM/TRM.txt", "w") as f:
        for i in pages:
            # filter numbered superscripts or subscripts
            clean_text = i.filter(lambda obj: ((obj["object_type"] == "char" and obj["size"] >= 10) or (obj["object_type"] == "char" and obj["size"]<9 and obj["y0"]>100 and obj["text"] not in ["1","2","3","4","5","6","7","8","9","0"])))
            
            # don't include tables
            table_loc = i.find_tables(table_settings)
            top_table = [x.bbox[1] for x in table_loc]
            bottom_table = [x.bbox[3] for x in table_loc]
            ranges = [(top_table[i],bottom_table[i]) for i in range(len(top_table))]
            clean_text = clean_text.filter(lambda obj: obj["object_type"] == "char" and not any(obj["top"]>top and obj["top"] < bottom for top,bottom in ranges))
            
            page_text = clean_text.extract_text_lines(y_tolerance = 5)
            for table_loc in top_table:
                for i in range(len(page_text)-1):
                    line_loc = page_text[i]["bottom"]
                    next_line_loc = page_text[i+1]["bottom"]
                    if table_loc < next_line_loc and table_loc > line_loc:
                        page_text.insert(i+1, {"text": "table is here", "bottom": table_loc})
            f.writelines([p["text"] + "\n" for p in page_text])


text = open("Exercises/TRM/TRM.txt").readlines()

document = [[]]

# Convert 1d text document to 2d where each column is a page
i = 0
curPage = 0
for j in text:
    document[curPage].append(j)
    if j.startswith("2023 IL TRM v11.0 Vol. 3_") and i < 70:
        for x in range(i,70):
            document[curPage].append("") # adding empty strings to compensate for different # of lines per page
        curPage += 1
        i = 0
        document.append([])
    i+=1

document = [i[0:69] for i in document]
document = document[0:479]
document = np.transpose(document)


# Get TRM Sections
head_text = [re.findall("— " + "5.[0-9].[0-9][0-9]?" + " .*", i) for i in text]
head_text = [i[0][2:] for i in head_text if i != []]
TRM_sections = (list(set(head_text)))
TRM_sections = [re.search("[0-9]\.[0-9]\.[0-9][0-9]?",sec).group(0) for sec in TRM_sections]
TRM_sections.remove("5.3.18")
TRM_sections.sort()

def get_pages(section):
    return np.where(np.char.find(document[0], section+" ")> -1)[0]

# Generate Tables
def get_tables(section):
    
    section_pages = get_pages(section)

    for i in pages[section_pages[0]:section_pages[-1]]:
        i.objects["char"] = list(filter(lambda obj: ((obj["object_type"] == "char" and obj["size"] >= 10) or (obj["object_type"] == "char" and obj["size"]<9 and obj["y0"]>100 and obj["text"] not in ["1","2","3","4","5","6","7","8","9","0", ","])), i.objects["char"]))
        

    tables = [pages[i-1].extract_tables(table_settings) for i in section_pages]

    tables = [t for sub in tables for t in sub]
    var = []
    
    for table in tables:
        for row in range(len(table)):
            # Remove None items
            index = [i for (i, ele) in enumerate(table[row]) if ele is None or ele == "Nan"]
            if len(index) != 0:
                for i in index:
                    table[row][i] = ''
            for ind in range(len(table[row])):
                table[row][ind] = table[row][ind].lower()
                table[row][ind] = table[row][ind].replace("δ", "∆")
                table[row][ind] = table[row][ind].replace("_", "")

    return tables

# Extract variable names and formulas
def extract_variable_names(formula):
    """
    Extracts variable names from a formula, including those that start with % or ∆, and now also
    those composed of multiple words separated by a space. Ensures variables with a leading % or ∆
    are correctly identified and included, while 'where:' is specifically excluded if captured.
    """
    # Adjusted pattern to capture variables composed of multiple words separated by spaces
    pattern = r'(?!.*:)[%∆#]?[a-zA-Z_γηµ][%#a-zA-Z0-9_γηµ]*(?:\s[%#a-zA-Z0-9_γηµ]+)*'
    variable_names = set(re.findall(pattern, formula))
    variable_names.discard("where:")  # Exclude "where:" if captured
    return variable_names

def find_formulas_and_variables(text_lines):
    """
    Identifies and captures formulas and variables from a list of text lines.
    Variables are defined across multiple lines starting with an equal sign, capturing until
    no more such lines are found.
    """
    formula_patterns = {
        "^∆kwh.*=": False,
        "^∆kw.*=": False,
        "^∆therm.*=": False,
        "^∆water.*=": False
    }
    formulas = {}
    all_variables = set()

    current_formula = ""

    capturing_formula = False
    keywords = set(["where", "early replacement", "early replacment", "fuel switch"])
    for line in text_lines:
        if line.startswith("time of sale: "):
            line = line.replace("time of sale: ", "")
        # Start capturing a formula definition
        if any([re.search(x, line) for x in formula_patterns.keys()]) and line.split("=")[0].strip() not in formulas.keys():
            capturing_formula = line.split("=")[0].strip()
            current_formula = line.split("=")[1].strip()
            if any(op in line.strip() for op in "/+-*()[]"):
                formulas[capturing_formula] = current_formula
            else:
                formulas[capturing_formula] = ""
            if all([x.replace(".","").isnumeric() for x in extract_variable_names(formulas[capturing_formula])]):
                1
            else:
                all_variables.update(extract_variable_names(formulas[capturing_formula]))
                keywords.update([var + " =" for var in all_variables])
            # Excluding formulas with numbers only
            if all([x.replace(".","").isnumeric() for x in extract_variable_names(formulas[capturing_formula])]) and formulas[capturing_formula]!="":
                del formulas[capturing_formula]
                current_formula = ""
                capturing_formula = False
        # Include lines until a keyword is encountered
        elif (not any(line.startswith(keyword) for keyword in keywords)) and capturing_formula:
            if not any(op in line.strip() for op in "/+-*()[]"):
                formulas[capturing_formula] += ""
            else:
                if line.startswith("="):
                    formulas[capturing_formula] = line.split("=")[1].strip()
                else:
                    current_formula = line.split("=")[0].strip()
                    formulas[capturing_formula] += current_formula
                all_variables.update(extract_variable_names(formulas[capturing_formula]))
                keywords.update([var + " =" for var in all_variables])
        else:
            current_formula = ""
            capturing_formula = False
        

    variables = {}
    capturing_variable = None
    for line in text_lines:
        if line.startswith("1/"):
            line = line.replace("1/","")
        # Start capturing a variable definition
        if any([line.startswith(x) for x in all_variables]) and not any(op in line.split("=")[0].strip() for op in "/+-*()[]"):
            possible_variable = line.split("=")[0].strip()
            if possible_variable not in variables.keys() and possible_variable not in formulas.keys():
                capturing_variable = possible_variable
                if "=" in line:
                    variables[capturing_variable] = line.split("=", 1)[1].strip()
                else:
                    variables[capturing_variable] = ""
            else:
                capturing_variable = False
        # Continue capturing the variable definition across lines
        elif (line.split("=")[0].strip() == "" or "=" not in line) and not ("for example" in line or "where" in line or "table is here" in line) and capturing_variable:
            variables[capturing_variable] += " " + line.strip()
        # Stop capturing the variable definition
        else:
            capturing_variable = None
    
    return formulas, variables


def get_equations(section):
    section_pages = get_pages(section)
    content = np.array([i[section_pages] for i in document]).transpose().flatten()
    
    #fix detected typos
    content = [x.lower().replace("δ", "∆") for x in content if x!=""]
    content = [x.lower().replace("×", "*") for x in content if x!=""]
    content = [x.lower().replace("", "∆") for x in content if x!=""]
    content = [x.lower().replace("_ consumption", "_consumption") for x in content if x!=""]
    content = [x.lower().replace("_ reduction", "reduction_") for x in content if x!=""]
    content = [x.lower().replace("_", "") for x in content if x!=""]
    content = [x.lower().replace("–", "-") for x in content if x!=""]
    content = [x.lower().replace("pre-", "pre") for x in content if x!=""]
    content = [x.lower().replace("side-by-side", "sidebyside") for x in content if x!=""]
    content = [x.lower().replace("btu/hr", "btuh") for x in content if x!=""]
    content = [x.lower().replace("btu/h", "btuh") for x in content if x!=""]
    content = [x.lower().replace("cooling", "cool") for x in content if x!=""]
    content = [x.lower().replace("cfssp", "cf") for x in content if x!=""]
    content = [x.lower().replace("- as calculated", "as calculated") for x in content if x!=""]
    content = [x.lower().replace("uefbaseline", "uefbase") for x in content if x!=""]
    content = [x.lower().replace("%naturalgas", "%fossil") for x in content if x!=""]
    content = [x.lower().replace("%gas", "%fossil") for x in content if x!=""]
    content = [x.lower().replace(",", "") for x in content if x!=""]
    content = [x.lower().replace("effcient", "efficient") for x in content if x!=""]
    content = [x.lower().replace("∆water (gallons)", "∆water") for x in content if x!=""]
    content = [x.lower().replace("∆therm ", "∆therms ") for x in content if x!=""]
    content = [x.lower().replace("/day", "_per_day") for x in content if x!=""]
    content = [x.lower().replace("/year", "_per_year") for x in content if x!=""]
    content = [x.lower().replace("/yr", "_per_year") for x in content if x!=""]
    content = [x.lower().replace("algorithms", "algorithm") for x in content if x!=""]
    content = [x.lower().replace("l/kwh", "l_kwh") for x in content if x!=""]
    content = [x.lower().replace("savings factor", "sf") for x in content if x!=""]
    content = [x.lower().replace("proportion of primary appliances", "primary usage") for x in content if x!=""]
    content = [x.lower().replace("/tree", "_per_tree") for x in content if x!=""]
    content = [x.lower().replace("/sqft", "_per_sqft") for x in content if x!=""]
    content = [x.lower().replace("celing/attic", "ceiling_attic") for x in content if x!=""]
    content = [x.lower().replace("ton-hr", "ton_hr") for x in content if x!=""]
    content = [x.lower().replace("kwh heatingelectric", "kwhheatingelectric") for x in content if x!=""]
    content = [x.lower().replace("kwhheating electric", "kwhheatingelectric") for x in content if x!=""]
    content = [x.lower().replace("kwhheating furnace", "kwhheatingfurnace") for x in content if x!=""]
    content = [x.lower().replace("hdd60", "hdd") for x in content if x!=""]
    content = [x.lower().replace("cdd65", "cdd") for x in content if x!=""]
    content = [x.lower().replace("ewater total", "ewater") for x in content if x!=""]
    content = [x.lower().replace("(resistance or heat pump)", "") for x in content if x!=""]
    content = [x.lower().replace("btu/therm", "") for x in content if x!=""]
    content = [x.lower().replace("shrink-fit", "shrinkfit") for x in content if x!=""]
    content = [x.lower().replace("direct-installed", "direct installed") for x in content if x!=""]
    content = [x.lower().replace("gal./ft2", "gal._per_ft2") for x in content if x!=""]
    content = [x.lower().replace("kWh/million gallons", "kWh_million gallons") for x in content if x!=""]
    content = [x.lower().replace("1/efficient", "1/ηefficient") for x in content if x!=""]
    content = [x.lower().replace("capacitygshpcool", "capacitycool") for x in content if x!=""]
    
    excluded_substrings = ["2023 il trm v11.0", "illinois statewide technical"]
    content = [x.lower() for x in content if not any(substring.lower() in x for substring in excluded_substrings)]
    content = content[content.index("algorithm\n"):content.index(list(filter(re.compile("d o&mc a c.*|m c :.*").match,content))[0])]
    
    return find_formulas_and_variables(content)

def print_outputs(formulas, variables):
    for frml, definition in formulas.items():
        print(f"{frml}: {definition}")
    print("\n")
    for var, definition in variables.items():
        print(f"{var}: {definition}")
        for table in tables:
            if var in list(np.array(table).flatten()):
                print(table)

    print("\n")

start = time.time()
tables = []
page_nos = []
variables = [[],[],[]]
formulas = [[],[],[]]
for section in TRM_sections:
    # tables.extend(get_tables(section))
    # page_nos.extend(get_pages(section))
    # a=1
    formula_data, var_data = get_equations(section)
    formulas[0].extend(list(formula_data.keys()))
    formulas[1].extend(list(formula_data.values()))
    formulas[2].extend([section]*len(formula_data))


    variables[0].extend(list(var_data.keys()))
    variables[1].extend([x.replace("\n"," ") for x in list(var_data.values())])
    variables[2].extend([section]*len(var_data))

formulas = np.array(formulas).T
variables = np.array(variables).T

problems = [[],[],[]]
for f in formulas:
    for v_f in list(extract_variable_names(f[1])):
        if v_f not in [v[0] for v in variables if v[2]==f[2]] and v_f not in [v[0] for v in formulas if v[2]==f[2]]:
            problems[0].append(v_f)
            problems[1].append(f[0]+' = ' + f[1])
            problems[2].append(f[2])

problems = np.array(problems).T
correct_sections = sorted(list(set(TRM_sections) - set(problems[:,2])))
a=1

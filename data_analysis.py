import csv
import os 

def get_column_index(column_name: str) -> int:
    assert (len(column_name) <= 2)
    assert (len(column_name) > 0)

    if (len(column_name) == 1):
        first_letter = column_name[0]
        second_letter = None
    else:
        first_letter = column_name[1]
        second_letter = column_name[0]

    index = ord(first_letter) - ord('A')
    if second_letter:
        index = index + 26 * (ord(second_letter) - ord('A') + 1)
        
    return index

column_mapping = {
    'OP_CARRIER_AIRLINE_ID': get_column_index('H'),
    'ARR_DELAY_GROUP': get_column_index('AT')

}

def get_field_name(fieldname1, fieldname2):
    if not fieldname1:
        assert(False)

    f = fieldname1
    if fieldname2:
        f = f + "_" + fieldname2
    
    return f

def get_row_index(row, fieldname1, fieldname2):
    if not fieldname1:
        assert(False)

    f = row[fieldname1]
    if fieldname2:
        f = f + "_" + row[fieldname2]
    
    return f

def perform_analysis(filename, fieldname1, fieldname2):
    X = {}  
    with open(filename, "r") as csvfile:
        inputfile = csv.DictReader(csvfile)
        for row in inputfile:
            if row['ARR_DELAY_GROUP']:
                row_index = get_row_index(row, fieldname1, fieldname2)
                column_index = int(row['ARR_DELAY_GROUP'])
                if row_index in X:
                    X[row_index][column_index] += 1
                else:
                    Y = {}
                    for item in range(-2, 13):
                        Y[item] = 0
                    X[row_index] = Y
                    X[row_index][column_index] += 1


    fieldname = get_field_name(fieldname1, fieldname2)
    with open(fieldname + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        final_list = [fieldname]
        for sub_item in range(-2, 13):
            final_list.append(str(sub_item) + "A")
        writer.writerow(final_list)
        for item in X:
            final_list = [item]
            for sub_item in range(-2, 13):
                final_list.append(X[item][sub_item])
            writer.writerow(final_list)
    return 

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
file_relative_path = r"Data\Jan_2016.csv"
abs_file_path = os.path.join(script_dir, file_relative_path)
fieldname1 = "DEST_AIRPORT_ID"
fieldname2 = ""
perform_analysis(abs_file_path, fieldname1, fieldname2)
pass

            
            

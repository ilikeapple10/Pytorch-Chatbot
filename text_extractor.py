import codecs
import pandas as pd
import csv

# Splits each line of the file to create lines and conversations
def loadLinesAndConversations(fileName):
    data = pd.read_csv(fileName)
    specialChars = {"’": "'", "…":""}
    data.replace(specialChars, regex=True, inplace=True)
    lines = {}
    conversations = {}
    for i in range(len(data)):
        lineObj = {}
        convObj = {}
        lineObj["lineID"] = i
        lineObj["text"] = data["text"][i]
        lines[lineObj['lineID']] = lineObj    

    grouped = data.groupby('conversation')
    # Iterate over the groups and create conversation objects
    for conversation_id, group in grouped:
        # Extract fields for conversation object
        convObj = {}
        convObj["conversationID"] = conversation_id
        convObj["lines"] = []

        # Iterate over the lines in the group and create line objects
        for index, row in group.iterrows():
            lineObj = {}
            lineObj["lineID"] = index
            lineObj["text"] = row["text"]
            convObj["lines"].append(lineObj)
            lines[lineObj["lineID"]] = lineObj

        conversations[convObj["conversationID"]] = convObj

    return lines, conversations

# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations.values():
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

def write_to_file(datafile, corpus):
    # Define path to new file
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict and conversations dict
    lines = {}
    conversations = {}
    # Load lines and conversations
    print("\nProcessing corpus into lines and conversations...")
    lines, conversations = loadLinesAndConversations(corpus)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)


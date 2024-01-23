
import json
import argparse


parser = argparse.ArgumentParser(
                    prog='Update distributedInfo',
                    description='Add distributedInfo into JSON trace')
parser.add_argument('filename') 
args = parser.parse_args()

TRACING_FILE = args.filename
f = open(TRACING_FILE)

data = json.load(f)

d = dict()
d["rank"] = 0

if "distributedInfo" not in data:
    data["distributedInfo"] = d
    print("distributedInfo added.")
else:
    print("Rank exists. Aborting.")
    exit()
with open(TRACING_FILE, "w") as f:
    json.dump(data, f)



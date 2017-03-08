import json
from datetime import datetime

print(datetime.now())
# data_file_name = 'Electronics_5.json'  # 13 secs loading time
data_file_name = 'Digital_Music_5.json'  # 1 sec loading time

f = open(data_file_name, 'r')
for line in f.readlines():
    json.loads(line)

f.close()
print(datetime.now())

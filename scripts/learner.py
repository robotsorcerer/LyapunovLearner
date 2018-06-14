

# arr = []
# with open ("moplan_data.csv","r") as inp:
#     #read line into array
#     for line in inp.readlines():
#         # loop over the elemets, split by whitespace
#         for i in line.split():
#             # convert to integer and append to the list
#             arr.append(int(i))

# arr = []
# with open ("moplan_data.csv","r") as inp:
#     #read line into array
#     for line in inp.readlines():
#         # add a new sublist
#         arr.append([])
#         # loop over the elemets, split by whitespace
#         for i in line.split():
#             # convert to integer and append to the last
#             # element of the list
#             arr[-1].append(float(i))

arr = []
with open ("moplan_data.csv","r") as inp:
    #read line into array
    for line in inp.readlines():
        if line.strip():
            [float(next(inp).strip()) for _ in range(4)]
        # # add a new sublist
        # arr.append([])
        # # loop over the elemets, split by whitespace
        # for i in line.split():
        #     # convert to integer and append to the last
        #     # element of the list
        #     arr[-1].append(float(i))

ß
print(arr)

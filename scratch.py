

ip = "Mr Jhon Smith     "

op = "Mr@20Jhon%20Smith"

print("=====",ip.strip(),"====")

st = ip.strip().split(" ")
ans = ""
for a in range (0,len(st)):
    if a <len(st)-1:
        ans+=st[a]+"%20"
    else:
        ans += st[a]


print(ans)

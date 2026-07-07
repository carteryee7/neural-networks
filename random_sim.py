import random


const = random.randint(1,6)

is_random = True

rolls = int(input("rolls: "))

count = 0

for i in range(rolls):
    roll = random.randint(1,6)

    if is_random:
        num = random.randint(1,6)
    else:
        num = const

    roll3 = random.randint(1,6)

    if roll == num == roll3:
        count += 1

print(count)
print(count/rolls)
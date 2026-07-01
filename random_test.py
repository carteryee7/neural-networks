import random

rolls = int(input("rolls: "))
sum = 0

low, high = int(input("min roll: ")), int(input("max roll: "))

for i in range(rolls):
    sum += random.randint(low, high)

print(f"avg of rolls: {sum / rolls}")
print(f"avg of range: {(low + high) / 2}")

# testing average of all the rolls is equal to the average of the range of the possible numbers
# converges to the average as the number of rolls increases
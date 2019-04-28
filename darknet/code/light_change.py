def change(number):
	t1=(10-number[0][0])*3
	if number[0][1]>=10:
		number[0][1]=10
	t2=10-number[0][1]
	print(float(t1+t2)*number[1])
	return float(t1+t2)*number[1]

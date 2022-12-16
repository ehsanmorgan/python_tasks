
def mul(x,y,z,r):
      for m in range (x,y):
            print('-------------------------------')
            
            for s in range(z,r):
                  print (f'{m}X{s}={m*s}')
                  print('--------------')
                  
                 
                  
mul(2,8,5,12)














numbers=[ x for x in range (1,101) if x%2 ==0]
print (numbers )


print(list(x for x in range (1,101) if x%2 ==0))

print(list(x for x in range (0,101,2) ))



def mul ():
      for x in range (1,101):
            if x%2==0 :
                  print(x)
                  
mul()




n=[True if x%2 ==0  else False for x in range(1,101) ]


def test (*names):
            names1=[]
            for x in names:
                  if len(x)>3 :
                        names1.append(x)
            print(list(names1))






      
test('ahmad','ali','mahmoud','nour')
















































print(n)

def mul(x,y,z,k):
      for m in range(x,y):
            print('----------------')
            for n in range(z,k):
                  print(f'{m} X {n}={m*n}')
                  
                  
                      

                
mul(2,12,8,14)





def ehsan(*names):
      names1=[]
      for x in names:
            if len(x)>4:
                  names1.append(x)
      print(names1)

ehsan('ehsan','ahmad','ali','mohamed','hasan')

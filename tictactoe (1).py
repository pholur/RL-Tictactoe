# Python code for Tic-Tac-Toe Machine Learning Tool
# Contributors: Pavan Holur, Ranjan P

# Python Version: 3.5.x

# https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
# https://www.packtpub.com/mapt/book/big_data_and_business_intelligence/9781786464453/7/ch07lvl1sec41/implementing-a-python-tic-tac-toe-game
# https://github.com/rolyatmax/tictactoe
# https://www.cs.swarthmore.edu/~meeden/cs63/f11/lab6.php
# https://en.wikipedia.org/wiki/Q-learning
# http://mnemstudio.org/path-finding-q-learning-tutorial.htm

# Use Q-Learning to Create a Tic-Tac Toe Game with different levels of training
# https://users.auth.gr/kehagiat/Research/GameTheory/12CombBiblio/TicTacToe.pdf

import numpy as np
import pickle
from copy import deepcopy
from operator import itemgetter

epochs = 200000
alpha = 0.3
k = 2*0.3/epochs
prob = 0.9
##############################################

def newGame():
	tic = np.array([[-10,-10,-10],[-10,-10,-10],[-10,-10,-10]])
	StateChainA = []
	StateChainB = []
	return tic, StateChainA, StateChainB

##############################################

def rot(mat):
	min1 = 10000
	So = 10000
	
	for i in range(0,4):
		min2,S = Find_Sx_So(np.rot90(mat,i))
		
		if(min2<min1):
 	     		min1 = min2
 	     		So = S
		elif(min2==min1) and (S<So):
 	     		So = S
      	
	return min1,So	
  
def findMin(mat):
	

  	
	min1,So1=rot(mat)
	min2,So2=rot(np.flip(mat,1))

	if(min2<min1):
		min1 = min2
		So1 = So2
	elif(min2==min1) and (So2<So1):
 	     	So1 = So2

  	
	return min1,So1

##############################################

def Find_Sx_So(currentState):
	
	Row_Major = currentState.flatten()
  
	Sx = 0
	So = 0
	for i in range(len(Row_Major)):
		if Row_Major[i] == 0:
			Sx = Sx + (0.5**(i+1))*0
			So = So + (0.5**(i+1))*1
    		# Solving for So
		elif Row_Major[i] == 1:
      			So = So + (0.5**(i+1))*0
      			Sx = Sx + (0.5**(i+1))*1
	return Sx,So

##############################################


def Check_Who_Won(CurrentState):
	# Flag default is 0
 	# return 1 when A wins and 2 when B wins and 3 when draw
	diag1 = CurrentState[1][1] + CurrentState[2][2] +CurrentState[0][0]
	diag2 = CurrentState[0][2] + CurrentState[1][1] +CurrentState[2][0]

	flag = 0 
	count = 0

	    
	for i in range(3):
		if(sum(CurrentState[i])==3) or (sum(CurrentState[:,i])==3) :
			flag = 1
		elif(sum(CurrentState[i]) == 0) or (sum(CurrentState[:,i]) == 0):
	      		flag = 2
  	
    
	
    
	if(diag1 == 3) or (diag2 == 3):
  		flag = 1
	elif(diag1 == 0) or (diag2 == 0):
		flag = 2

	for i in CurrentState.flatten():
		if(i==-10):
			count += 1
	if(not count) and (not flag):
		return 3
    
	return flag
   


##############################################
def UpdateMapA(mapA, StateChainA, Winner):
	global alpha
	global k
	Ps = 0
	try:
		temp = np.array(mapA)[:,0:2]
	except:
		temp = np.array(mapA)

	if Winner == 0:
		pass
	elif (Winner == 1 or Winner == 3):
		if StateChainA[-1] not in temp.tolist():
			mapA.append([StateChainA[-1][0], StateChainA[-1][1], 1])
		Ps = 1
		for Sx, So in StateChainA[-2::-1]:
			if [Sx,So] in temp.tolist():
				pTurd = temp.tolist().index([Sx,So])
				mapA[pTurd][2] = mapA[pTurd][2] + alpha*(Ps - mapA[pTurd][2])	
				Ps = mapA[pTurd][2]
			else:
				mapA.append([Sx, So, (0.5 + alpha*(Ps - 0.5))])
				Ps = 0.5 + alpha*(Ps - 0.5)
        
	elif (Winner == 2):
		if StateChainA[-1] not in temp.tolist():
			mapA.append([StateChainA[-1][0], StateChainA[-1][1], 0])
		Ps = 0
		for Sx, So in StateChainA[-2::-1]:
			if [Sx,So] in temp.tolist():
				pTurd = temp.tolist().index([Sx,So])
				mapA[pTurd][2] = mapA[pTurd][2] + alpha*(Ps - mapA[pTurd][2])
				Ps = mapA[pTurd][2]
			else:
				mapA.append([Sx, So, (0.5 + alpha*(Ps - 0.5))])
				Ps = 0.5 + alpha*(Ps - 0.5)
	return mapA                                       
      
    	
def UpdateMapB(mapB, StateChainB, Winner):
	global alpha
	global k
	Ps = 0
	try:
		temp = np.array(mapB)[:,0:2]
	except:
		temp = np.array(mapB)
	
	if Winner == 0:
		pass
	elif (Winner == 2 or Winner == 3):
		if StateChainB[-1] not in temp.tolist():
			mapB.append([StateChainB[-1][0], StateChainB[-1][1], 1])
		Ps = 1
		for Sx, So in StateChainB[-2::-1]:
			if [Sx,So] in temp.tolist():
				pTurd = temp.tolist().index([Sx,So])
				mapB[pTurd][2] = mapB[pTurd][2] + alpha*(Ps - mapB[pTurd][2])
				Ps = mapB[pTurd][2]
			else:
				mapB.append([Sx, So, (0.5 + alpha*(Ps - 0.5))])
				Ps = 0.5 + alpha*(Ps - 0.5)
        
	elif (Winner == 1):
		if StateChainB[-1] not in temp.tolist():
			mapB.append([StateChainB[-1][0], StateChainB[-1][1], 0])
		Ps = 0
		for Sx, So in StateChainB[-2::-1]:
			if [Sx,So] in temp.tolist():
				pTurd = temp.tolist().index([Sx,So])
				mapB[pTurd][2] = mapB[pTurd][2] + alpha*(Ps - mapB[pTurd][2])
				Ps = mapB[pTurd][2]
			else:
				mapB.append([Sx, So, (0.5 + alpha*(Ps - 0.5))])
				Ps = 0.5 + alpha*(Ps - 0.5)
	return mapB     
        
##############################################
def MoveA(CurrentState, mapA):
	Prob_List = []
	global prob
	#print "MovA: CurrentState = " + str(CurrentState)
	for i in range(3):
		for j in range(3):
			if CurrentState[i][j] == -10:
				tempCurrentState = deepcopy(CurrentState)
				tempCurrentState[i][j] = 1
				Sx,So= findMin(tempCurrentState)
				if(len(mapA)!=0):        			
					if [Sx,So] not in np.array(mapA)[:,0:2].tolist():
						Prob_List.append((i,j,0.5,Sx,So))
					else:
                                                
			  			Prob_List.append((i,j,mapA[np.array(mapA)[:,0:2].tolist().index([Sx,So])][2],Sx,So))
				else:
					Prob_List.append((i,j,0.5,Sx,So))
					
  
	if np.random.rand() < prob:
	  	tempStateDet = max(Prob_List,key = itemgetter(2))
	else:
	  	tempStateDet= Prob_List[int(np.random.rand()*(len(Prob_List)))]
 	
	#print Prob_List
	#print "Temp:" + str(tempStateDet)
	CurrentState[tempStateDet[0]][tempStateDet[1]] = 1
	StateChainA = [tempStateDet[3],tempStateDet[4]]
	
	return CurrentState, StateChainA


def MoveB(CurrentState, mapB):
	Prob_List = []
	global prob
	#print "MovB: CurrentState = " + str(CurrentState)

	for i in range(3):
		for j in range(3):
			if CurrentState[i][j] == -10:
				tempCurrentState = deepcopy(CurrentState)
				tempCurrentState[i][j] = 0
				Sx,So= findMin(tempCurrentState)
				if(len(mapB)!=0): 
				      			
					if ([Sx,So] not in np.array(mapB)[:,0:2].tolist()):
		  				Prob_List.append((i,j,0.5,Sx,So))
					else:	
						
						Prob_List.append((i,j,mapB[np.array(mapB)[:,0:2].tolist().index([Sx,So])][2],Sx,So))
				else:
					Prob_List.append((i,j,0.5,Sx,So))	
  
	if np.random.rand() < prob:
  		tempStateDet = max(Prob_List,key = itemgetter(2))
	else:
  		tempStateDet= Prob_List[int(np.random.rand()*(len(Prob_List)))]
    
	CurrentState[tempStateDet[0]][tempStateDet[1]] = 0
	StateChainB = [tempStateDet[3],tempStateDet[4]]
	return CurrentState, StateChainB
   
##############################################  
if __name__ == "__main__":
	global prob
	mapA=[]
	mapB=[]
	Acount = 0
	Bcount = 0
	DrawCount = 0
	z = 0
	try:
		f = open('mapA.pkl','rb')
		mapA = pickle.load(f)
		print ("Already trained.")
		print ("Entering Game Mode...")
		prob = 1
		CurrentState,_,_ = newGame()
		ch = input("Do you want to play first? (y/n): ")
		
		while(True and ch == 'n'):

                        
		        CurrentState,_ = MoveA(CurrentState,mapA)
		        print (CurrentState)
                        
		        temp = Check_Who_Won(CurrentState)
		        if (temp == 1):
			        print ("You Lose!")
			        break
		        elif (temp == 3):
			        print ("Draw")
			        break
                        
		        move = input("Enter the Co-ordinates of the move(00 ---> 22): ")
		        CurrentState[int(move[0]),int(move[1])] = 0
		        print (CurrentState)
                                         
		        temp = Check_Who_Won(CurrentState)
		        if (temp == 2):
			        print ("You Win!")
			        break
		        elif (temp == 3):
			        print ("Draw")
			        break

                        
		while(True and ch == 'y'):
			print (CurrentState)
			move = input("Enter the Co-ordinates of the move(00 ---> 22): ")
			CurrentState[int(move[0]),int(move[1])] = 0
			temp = Check_Who_Won(CurrentState)
			if(temp == 2):
				print ("You Win!")
				break
			elif temp ==3:
				print ("Draw")
				break
			CurrentState,_ = MoveA(CurrentState,mapA)
			print (CurrentState)
			temp = Check_Who_Won(CurrentState)
			if(temp==1):
			        print ("You Lose!")
			        break
			elif(temp==3):
				print ("Draw")
				break
	except:
		prob = 0.9
		print (prob)
		print ("Training in Progress...")
		for i in range(epochs): # Number of games to be played
			flag = 0
			
			print ("Playing game " + str(i+1) + " of " + str(epochs))
			CurrentState, StateChainA, StateChainB = newGame()
			# A will always start first       
			while (flag == 0):
				if(i%2)==0:
					CurrentState,x = MoveA(CurrentState, mapA)
					StateChainA.append(x)
					temp = Check_Who_Won(CurrentState)
					if temp:
						flag=temp
						break
					CurrentState,x = MoveB(CurrentState,mapB)
					StateChainB.append(x)
					
					temp = Check_Who_Won(CurrentState)
					#print CurrentState
					if temp:
						flag = temp
						break
					flag = Check_Who_Won(CurrentState)
				else:
					CurrentState,x = MoveB(CurrentState, mapB)
					StateChainB.append(x)
					temp = Check_Who_Won(CurrentState)
					if temp:
						flag=temp
						break
					CurrentState,x = MoveA(CurrentState,mapA)
					StateChainA.append(x)
				
					temp = Check_Who_Won(CurrentState)
					#print CurrentState
					if temp:
						flag = temp
						break
					flag = Check_Who_Won(CurrentState)

			if flag == 1:
				mapA = UpdateMapA(mapA, StateChainA, 1)
				mapB = UpdateMapB(mapB, StateChainB, 1)
				Acount+=1
				print ("Game " + str(i) + " won by A")
			elif flag == 2:
				mapA = UpdateMapA(mapA, StateChainA, 2)
				mapB = UpdateMapB(mapB, StateChainB, 2)
				Bcount+=1
				print ("Game " + str(i) + " won by B")
			elif flag == 3:
				mapA = UpdateMapA(mapA, StateChainA, 3)
				mapB = UpdateMapB(mapB, StateChainB, 3)
				DrawCount+=1
				print ("Draw")
			

			if((i%2)==0):
				alpha = alpha - k				
			'''
			print mapA
			print mapB
			'''
		pickle.dump(mapA,open('mapA.pkl','wb'))
		pickle.dump(mapB,open('mapB.pkl','wb'))
		#print pickle.load(open('mapA.pkl','rb'))

		print ("Wins: A- " + str(Acount))
		print ("Wins: B- " + str(Bcount))
		print ("Wins: Draw- " + str(DrawCount))
		
		
	
	
	

		
				
		

  





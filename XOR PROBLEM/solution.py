import numpy as np 
import math 
import matplotlib.pyplot as plt
# IMPORT THE DATA
X = np.array([(0,0),(0,1),(1,0),(1,1)])
Y = np.array([0,1,1,0])

N2 , D2 = X.shape   # D2 is the number of entries per iteration columns

#Adding the bias 

X = np.append(X,np.ones((N2,1))*-1,axis=1)


#Hidden layer
Nhl  = 2
Number_weights_per_neurom = D2+1 # The bias weight is taking care with the 1
weights_hidden_layer = Number_weights_per_neurom*Nhl
Whl = np.random.rand(Nhl,Number_weights_per_neurom)   # matrix   2x 3
Output = np.zeros((Nhl,1))
Sensitividad_hl = np.zeros(((Nhl,1) ))
Sensitividad2_hl = np.zeros(((Nhl,1) ))
Senns_hl = np.zeros(((Nhl,1) ))
dp_hl = np.zeros((D2,Number_weights_per_neurom),dtype=float) 

#Output layer
Nol = 1
Number_weights_per_neurom_ol = Nhl+1
weights_output_layer = Number_weights_per_neurom_ol*Nol
Wol =   np.random.rand(Nol,Number_weights_per_neurom_ol)
Outpu_ol = np.zeros(1)
Sensitividad_ol = np.zeros(((Nol,1) ))
Sensitividad2_ol = np.zeros(((Nol,1) ))
Senns_ol = np.zeros(((Nol,1) ))
dp_ol = np.zeros((Nol,Number_weights_per_neurom_ol))   # delta weith
error_ol =  np.zeros(((Nol,1) ))

alpha = 0.1 #learning factor

#Training

error_d = 0.01 #Desired error
E_p = 100 #Actual error
E_total =[]
Error_iteracion = 0
epocas = 0
#Una epoca manual probar
X2 = "a"
#while X2 != "b":
while E_p >= error_d:
	#X2 = input("Continuamos?")
	Error_iteracion = 0  # clean the error on each iteration
	#for para cambiar cada iteracion 
	for i in range(0,N2):
		#Recorrido hacia adelante 
		#Output hidden layer 
		#aux1 = Whl[0,0]*X[i,0]  +Whl[0,1]*X[i,1]   +  Whl[0,2]*X[i,2]
		#aux2 = Whl[1,0]*X[i,0]  +Whl[1,1]*X[i,1]   +  Whl[1,2]*X[i,2]
		aux =  np.sum(Whl*X[i,:],axis=1)

		#Output[0] =  1/(1+np.exp(-aux[0]))
		#Output[1] =  1/(1+np.exp(-aux[1]))
		Output =  1/(1+np.exp(-aux))
		Output = Output.reshape((-1,1))
		#print(Output)

		


		#Output last layer calculate the entries for the last later
		Xol = []
		Xol = np.append(Output,np.ones((Nol,1))*-1 , axis=0)

		aux3 = Wol[0,0]*Xol[0]  + Wol[0,1]*Xol[1] + Wol[0,2]*Xol[2] 
		#aux3 =np.sum(Wol*Xol)
		
		Outpu_ol = 1/(1+ np.exp(-aux3))

		#print("Salida " +  str(Outpu_ol) +    " deseado  " + str(Y[i]))

		#Error
		error_ol = Y[i] - Outpu_ol
		Error_iteracion = Error_iteracion + error_ol*error_ol
		#print(error_ol)

		###############Backpropagtion weight correction #######################

		#sensitiviti output layer 
		Sensitividad_ol = Outpu_ol*(1-Outpu_ol)*error_ol
		#print(Sensitividad_ol)

		#calculate the Weith adjustment output layer

		dp_ol[0,0] = alpha*Xol[0,0]*Sensitividad_ol
		dp_ol[0,1] = alpha*Xol[1,0]*Sensitividad_ol
		dp_ol[0,2] = alpha*Xol[2,0]*Sensitividad_ol
		#dp_ol = alpha*Xol*Sensitividad_ol
		



		#Sensitiviti hidden layer
		Sensitividad_hl[0,0] = Output[0]*(1-Output[0])*Sensitividad_ol*Wol[0,0]
		Sensitividad_hl[1,0] =  Output[1]*(1-Output[1])*Sensitividad_ol*Wol[0,1]
		#calculate the Weith adjustment hiddem  layer
		dp_hl[0,0] = alpha*X[i,0]*Sensitividad_hl[0,0]
		dp_hl[0,1] = alpha*X[i,1]*Sensitividad_hl[0,0]
		dp_hl[0,2] = alpha*X[i,2]*Sensitividad_hl[0,0]

		dp_hl[1,0] = alpha*X[i,0]*Sensitividad_hl[1,0]
		dp_hl[1,1] = alpha*X[i,1]*Sensitividad_hl[1,0]
		dp_hl[1,2] = alpha*X[i,2]*Sensitividad_hl[1,0]

		# weith adjustment 
		#hidden layer
		Whl[0,0] = Whl[0,0] + dp_hl[0,0] 
		Whl[0,1] = Whl[0,1] + dp_hl[0,1] 
		Whl[0,2] = Whl[0,2] + dp_hl[0,2] 

		Whl[1,0] = Whl[1,0] + dp_hl[1,0] 
		Whl[1,1] = Whl[1,1] + dp_hl[1,1] 
		Whl[1,2] = Whl[1,2] + dp_hl[1,2] 

		#outpu layer
		#Wol = Wol + dp_ol
		Wol[0,0] = Wol[0,0] + dp_ol[0,0] 
		Wol[0,1] = Wol[0,1] + dp_ol[0,1] 
		Wol[0,2] = Wol[0,2] + dp_ol[0,2] 
		#END iteration
	
	E_p = math.sqrt(Error_iteracion)/N2
	E_total.append(E_p)
	print(E_p)
	#print(epocas)
	epocas+=1
	#end epoca

#prueba de que funciona la red
print("Entrenamiento finalizado......Se demoro "+ str(epocas))
ax= plt.subplot()
Epocas = range(epocas)
ax.plot(Epocas, E_total)
ax.set(xlabel='Epocs', ylabel='Cuadratic error',
       title='Neural net error XOR problem')
ax.grid()


plt.show()

print("Inicia test")
for i in range(0,N2):
	#Recorrido hacia adelante 
	#Output hidden layer 
	aux1 = Whl[0,0]*X[i,0]  +Whl[0,1]*X[i,1]   +  Whl[0,2]*X[i,2]
	aux2 = Whl[1,0]*X[i,0]  +Whl[1,1]*X[i,1]   +  Whl[1,2]*X[i,2]

	Output[0] =  1/(1+np.exp(-aux1))
	Output[1] =  1/(1+np.exp(-aux2))

	#print(Output)


	#Output last layer
	Xol = []
	Xol = np.append(Output,np.ones((Nol,1))*-1 , axis=0)

	aux3 = Wol[0,0]*Xol[0]  + Wol[0,1]*Xol[1] + Wol[0,2]*Xol[2] 
	Outpu_ol = 1/(1+ math.exp(-aux3))

	print("Salida " +  str(Outpu_ol) +    " deseado  " + str(Y[i]))

	#Error
	error_ol = Y[i] - Outpu_ol
	Error_iteracion = Error_iteracion + error_ol*error_ol
	#print(error_ol)








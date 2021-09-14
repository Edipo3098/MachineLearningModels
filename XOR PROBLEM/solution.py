import numpy as np 

# IMPORT THE DATA
X = np.array([(0,0),(0,1),(1,0),(1,1)])
Y = np.array([0,1,1,0])




N2 , D2 = X.shape   # D2 is the number of entries per iteration
#Adding the bias weight

X = np.append(X,np.ones((N2,1)),axis=1)
print(X)

#Hidden layer
Nhl  = 2
Number_weights_per_neurom = D2+1
weights_hidden_layer = Number_weights_per_neurom*Nhl
Whl = np.zeros((D2,Number_weights_per_neurom),dtype=float) 
Output = np.zeros((Nhl,1))
Sensitividad_hl = np.zeros(((Nhl,1) ))
Sensitividad2_hl = np.zeros(((Nhl,1) ))
Senns_hl = np.zeros(((Nhl,1) ))
dp_hl = np.zeros(((Nhl,1) ))

#Output layer
Nol = 1
Number_weights_per_neurom_ol = Nhl+1
weights_output_layer = Number_weights_per_neurom_ol*Nol
Wol =  np.zeros((Nol,Number_weights_per_neurom))
Outpu_ol = np.zeros(1)
Sensitividad_ol = np.zeros(((Nol,1) ))
Sensitividad2_ol = np.zeros(((Nol,1) ))
Senns_ol = np.zeros(((Nol,1) ))
dp_ol = np.zeros(((Nol,1) ))


#Training

error_d = 0.01 #Desired error
E_p = 100 #Actual error
while(error_d <= E_p):
    break

    #Front propagation
    for i in range(0,3):
        break





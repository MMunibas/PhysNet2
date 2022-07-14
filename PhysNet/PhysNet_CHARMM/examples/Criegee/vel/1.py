#mom650
import numpy as np
momH=[.04648520,2.36518304,.70508912]
momO=[4.124645320,-17.080959440,2.362791392]
#com
com=[139.8796,-493.1833,102.9019]
comH=[139.22780-139.8796,(-493.65314+493.1833),(103.32618-102.9019)]
comO=[139.92062-139.8796,(-493.15372+493.1833),(102.87518-102.9019)]

angmomH= np.cross(momH,comH)
angmomO= np.cross(momO,comO)
print(angmomH)
print(angmomO)

print(angmomH[0]+ angmomO[0], angmomH[1]+ angmomO[1], angmomH[2]+ angmomO[2])




#mom649
momH1=[-1.823416,1.146803,-1.321829]
momO1=[5.998943,-15.851031,4.397201]
#com
com1=[138.8767,-489.6455,102.1639]
comH1=[138.17644 -com1[0],(-490.15762 -com1[1]),(102.21863 -com1[2])]
comO1=[138.92087 -com1[0],(-489.61325-com1[1]),( 102.16041-com1[2])]

angmomH1= np.cross(momH1,comH1)
angmomO1= np.cross(momO1,comO1)
print(angmomH1)
print(angmomO1)

print(angmomH1[0]+ angmomO1[0], angmomH1[1]+ angmomO1[1], angmomH1[2]+ angmomO1[2])










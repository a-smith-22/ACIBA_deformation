'''
TITLE: vSteps.py script
AUTHORS: Written by Z. Hossain and A. Smith.
LATEST VERSION: 08-18-2022
ABOUT: Compute the displacement of each node (CS, VS) in an input exodus file (.GEN) and create the corresponding output exodus file at each value in time.
UPDATES:
08-15-2022 V1.0 -> Initial version
08-15-2022 V2.0 -> Only one bump is sampled for displacement, radial symmetry is applied.
08-16-2022 V2.1 -> Optimized arclength computation, introduced interpolation to thickness calculations.
08-16-2022 V2.2 -> Reduced number of needed samples to reduce computation time.
08-17-2022 V2.3 -> Fixed boundary displacement discrepency at y=0 values.
08-17-2022 V3.0 -> Fix error, displacement is the true difference of points.
08-17-2022 V3.1 -> Increased accuracy of closesnt point arclength calculation by adding corrective term. 
'''
## Run the following commands in order: 
## vpkg_require exodus/20200218
## vpkg_require python-envs/exodus-example:20210104
## python vSteps-ACIBA.py -i aciba.gen -o aciba_out.gen --time_min 0 --time_max 1 --time_numstep 10 --E 1.0 --nu 0.33 --ampl 1.0 --initialpos 0 0 0 --cs 0001 --strainRate 1.0 --slope 1.0

#!/usr/bin/env python

print(' ')

import numpy as np
import time
import os
os.system('rm aciba_out.gen')
import exodus3 as exo



#========================================DEFINING PARAMETERS========================================



#ACIBA parameters
R2 = 0.057323440        #inital radius (cubit)
h2 = 0.008000000        #initial bump height (cubit)
R1 = 0.0830665059443    #final radius (excel)
h1 = 0.003000000        #final bump height (excel)
T = 0.0008     #total thickness of membrane (constant)
n = 12         #number of bumps (constant)

#Computation parameters
mesh_size = 0.00015               #size of meshing (cubit)
arclength = 0.545634857717140     #total arclength (excel)

smp_prct = 2.0 #number of samples as percentage of estimated number of nodes (recommended values >1.5)
samples = int( np.ceil( smp_prct * (arclength / n) / mesh_size ) ) #minimum number of samples
bmp_ang =  2*np.pi/n              #angle of each bump
dtheta = bmp_ang / samples        #angular distance between samples, only sample one section
dtheta_inv = samples / bmp_ang    #inverse of above, speeds computation by limiting division



print('>PROGRESS: ACIBA conditions read...')
print('...R1 = ' + str(R1))
print('...h1 = ' + str(h1))
print('...R2 = ' + str(R2))
print('...h2 = ' + str(h2))



#========================================INPUT ARGUMENTS========================================



#Recieve inputs of script from user. 
def parse(args=None):
    import argparse
    ### Get options from the command line
    parser = argparse.ArgumentParser(description='Compute boundary displacement for a surfing computation')
    parser.add_argument('-i','--inputfile',help='input file',default=None)
    parser.add_argument('-o','--outputfile',help='output file',default=None)
    parser.add_argument("--time_min",type=float,help="first time step",default=0.0)
    parser.add_argument("--time_max",type=float,help="last time step",default=1.0)
    parser.add_argument("--time_numstep",type=int,help="number of time step",default=10)
    parser.add_argument("--E",type=float,help="Youngs modulus",default=1.0)
    parser.add_argument("--nu",type=float,help="Poisson Ratio",default=0.33)
    parser.add_argument("--ampl",type=float,help="Amplification",default=1.0)    
    parser.add_argument("--initialpos",type=float,nargs=3,help="initial crack tip postion",default=[0.0,0.0,0.0])
    parser.add_argument("--cs",type=int,nargs='*',help="list of cell sets where surfing boundary displacements are applied",default=[])
    parser.add_argument("--vs",type=int,nargs='*',help="list of vertex sets where surfing boundary displacements are applied",default=[])
    parser.add_argument("--plasticity",default=False,action="store_true",help="Add extended variables for plasticity related fields")
    parser.add_argument("--force",action="store_true",default=False,help="Overwrite existing files without prompting")
    parser.add_argument("--strainRate",type=float,help="Strain rate imposed to translate the displacement field",default=1.0)
    parser.add_argument("--slope",type=float,help="slope of the nonzero displacement",default=1.0)
    return parser.parse_args()
print('>PROGRESS: Options from the command line are read...')    

#Parse plasticity options.
def exoformat(e,plasticity=False):
    if plasticity:
        global_variable_name = ["Elastic Energy","Work","Surface Energy","Total Energy","Dissipation Plastic"]
        if e.num_dimensions() == 2: 
            node_variable_name  = ["Temperature","Damage","Displacement_X","Displacement_Y"]
            element_variable_name   = ["External_Temperature","Heat_Flux","Pressure_Force",
                                       "Force_X","Force_Y",
                                       "Stress_XX","Stress_YY","Stress_XY",
                                       "Cumulated_Plastic_Energy","plasticStrain_XX","plasticStrain_YY","plasticStrain_XY"]
        else:
            node_variable_name  = ["Temperature","Damage","Displacement_X","Displacement_Y","Displacement_Z"]
            element_variable_name   = ["External_Temperature","Heat_Flux","Pressure_Force",
                                       "Force_X","Force_Y","Force_Z",
                                       "Stress_XX","Stress_YY","Stress_ZZ","Stress_YZ","Stress_XZ","Stress_XY",
                                       "Cumulated_Plastic_Energy","plasticStrain_XX","plasticStrain_YY","plasticStrain_ZZ","plasticStrain_XY","plasticStrain_YZ","plasticStrain_XZ","plasticStrain_XY"]
    else:
        global_variable_name = ["Elastic Energy","Work","Surface Energy","Total Energy"]
        if e.num_dimensions() == 2: 
            node_variable_name  = ["Temperature","Damage","Displacement_X","Displacement_Y"]
            element_variable_name   = ["External_Temperature","Heat_Flux","Pressure_Force",
                                       "Force_X","Force_Y",
                                       "Stress_XX","Stress_YY","Stress_XY"]
        else:
            node_variable_name  = ["Temperature","Damage","Displacement_X","Displacement_Y","Displacement_Z"]
            element_variable_name   = ["External_Temperature","Heat_Flux","Pressure_Force",
                                       "Force_X","Force_Y","Force_Z",
                                       "Stress_XX","Stress_YY","Stress_ZZ","Stress_YZ","Stress_XZ","Stress_XY"]
    e.set_global_variable_number(0)
    e.set_node_variable_number(len(node_variable_name))
    for i in range(len(node_variable_name)):
        e.put_node_variable_name(node_variable_name[i],i+1)
    e.set_element_variable_number(len(element_variable_name))
    for i in range(len(element_variable_name)):
        e.put_element_variable_name(element_variable_name[i],i+1)
    e.set_element_variable_truth_table([True] * e.numElemBlk.value * len(element_variable_name))
    return 0 
print('>PROGRESS: Options related to plasticity are read ...')  



#========================================DISPLACEMENT FUNCTIONS========================================



#I -> Neutral axis values of initial and final structure
def dist(x1,y1,x2,y2):
    '''Pythagorean distance between two cartesian points.'''
    delx = x2-x1
    dely = y2-y1
    return np.sqrt(delx**2 + dely**2)

def ds_aciba(R, h, num, ang, dang):
    '''Computes a tiny (dang) arclength component of the ACIBA strucutre (R, h, num) at a given angle (ang). Used to compute the overall
       arclength when the function is called many times and values summed.'''
    r = R + h * np.sin(num*ang)
    drdt = h * num * np.cos(num*ang)
    arg = np.sqrt( r**2 + drdt**2 )
    return arg*dang

def neutral_axis(R,h,num,s):
    '''Compute list of points on neutral axis of one bump of ACIBA structure. Input ACIBA parameters (R,h,num) as well as
       number of samples (s) to take and output X and Y values as list. Output cummulative sum of arclength up to point,
       used to map values later.'''
    '''Notes:
       -Values start at X[0]=R, Y[0]=0 (R,0) then work counter-clockwise.'''
    #Sample points on structure
    x_out = []
    y_out = [] #create blank lists
    for i in range(s+2): #add extra to compute boundaries 
        ang = (i-1)*dtheta #compute left reimann sum
        rad = R + h * np.sin(num*ang)
        x_out.append(rad * np.cos(ang))
        y_out.append(rad * np.sin(ang))
        
    #Calculate linear distance between sampled values
    s_out = []
    s_total = 0 #cummulative sum of arclength
    dang = bmp_ang / s #dtheta for arclength computation
    for i in range(s+2):
        ang = i*dtheta #arclength value applies to index after it
        s_total += ds_aciba(R1, h1, n, ang, dang)
        if i == 0: #skip first term          
            s_out.append(0) #add zero as first value
        s_out.append(s_total) #add to cummulative sum list
        
    return x_out, y_out, s_out
x1_vals, y1_vals, s1_vals = neutral_axis(R1,h1,n,samples)
x2_vals, y2_vals, s2_vals = neutral_axis(R2,h2,n,samples)
print('>PROGRESS: Neutral axis coordinates computed...')



#II -> Compute closest value from neutral axis
def rotate(x, y, ang):
    '''Rotate a point counter-clockwise around origin using matrix transformation. Output as new point coordinates.'''
    x_new = x * np.cos(ang) - y * np.sin(ang)
    y_new = x * np.sin(ang) + y * np.cos(ang)
    return x_new, y_new
        
def na1_pt_data(na_x, na_y, x, y):
    '''Find the location of the closest point on the neutral axis from the input point (x,y). Return the
       thickness (T) (distance) and arclength from NA-pt, and the angle to original point. '''
    '''Notes:
       -Lists na_x and na_y (x and y point data) are calculated from "def neutral_axis" function above. '''
    
    #Map input point to first section
    pt_ang = np.arctan2(y,x) #actual angle
    mod_ang = pt_ang % (bmp_ang) #angle in single section
    rot_ang = pt_ang - mod_ang #ccw angle to rotate about origin
    x,y = rotate(x, y, -rot_ang) #rotate clockwise, position in single section
    new_ang = np.arctan2(y,x) #angle of point in single section
    
    #Index of point
    i = int(np.floor( mod_ang * dtheta_inv ) ) #ranges from 0 (axis) to samples (end of section)
    
    #Compute distance to NA using vector projection
    #approx tangent line
    if i <= 0: #beginning region
        temp_r = R1 + h1 * np.sin(n*(i-1)*dtheta)
        temp_x = temp_r * np.cos((i-1)*dtheta)
        temp_y = temp_r * np.sin((i-1)*dtheta) #add point outside of x1_vals/y1_vals for boundary calculations
        tx, ty = (na_x[i+1]-temp_x), (na_y[i+1]-temp_y)
    elif i >= samples: #boundary region
        tx, ty = (na_x[i+1]-na_x[i-1]), (na_y[i+1]-na_y[i-1]) #use 2 point backward
    else: #mid-section
        tx, ty = (na_x[i+1]-na_x[i-1]), (na_y[i+1]-na_y[i-1]) #center point approx
    #find normal and position vector
    nx, ny = ty, -tx  #outwards facing normal
    n_mag_inv = (nx**2 + ny**2) ** -0.5     #inverse of the magnitude of normal vector
    qx, qy = (x - na_x[i]), (y - na_y[i])   #vector from na-pt to point
    T_min = (nx*qx + ny*qy) * n_mag_inv #distance to NA, + is for outward thickness
    
    #Compute true arclength
    s = s1_vals[i] #underapproximation
    j = mod_ang * dtheta_inv #true index of neutral axis point, middle point
    ang = j*dtheta #polar angle to this middle NA-pt
    dang = ang - i*dtheta #angular difference between underapprox and actual
    s_offset = ds_aciba(R1,h1,n,ang,dang)
    s += s_offset #add corrective term
    
    #return distance to NA-pt, arclength to NA=pt, and polar angle
    return T_min, s, pt_ang


    
    
#III -> Map neutral axis point on final structure
def mapped_pt(R, h, num, th, s, x_vals, y_vals):
    '''Map a point from the initial configuration to the final configuration. Input the final structure parameters (R,h,n) as well
       as the arclength (s) and thickness (th) to define the point. List of points (x_vals, y_vals) are needed to compute the
       arclength as well. Output the coordinates (X',Y') of the new point. Note:
       Positive thickness is radially further from the origin. 
       See cubit for calculation derivation.'''
    s2 = 0 #current arclength
    i = 0 #index
    while s2 <= s:
        s2 = s2_vals[i] #check value of cummulative sum
        i += 1
        if i > samples:
            break #exit clause
    #stop when arclength is more than cummulative sum. s and i will be >0
    
    #Linear interpolate point on neutral axis
    ang_und = (i-1)*dtheta
    r_a = R + h * np.sin(num*ang_und)
    x_a = r_a * np.cos(ang_und)
    y_a = r_a * np.sin(ang_und) #underapprox point
    
    ang = i*dtheta
    r_b = R + h * np.sin(num*ang)
    x_b = r_b * np.cos(ang)
    y_b = r_b * np.sin(ang) #overapprox point (point stopped at)
    
    m = (s - s2_vals[i-1]) / (s2_vals[i] - s2_vals[i-1]) #constant slope term
    x_out = (x_b - x_a) * m + x_a
    y_out = (y_b - y_a) * m + y_a
    ang = np.arctan2(y_out,x_out) #linear interpolate values for NA2-pt, calculate interpolated angle

    #Calculate offset from normal vector
    dxdt = h * num * np.cos(num*ang) * np.cos(ang) - R * np.sin(ang) - h * np.sin(num*ang) * np.sin(ang)
    dydt = h * num * np.cos(num*ang) * np.sin(ang) + R * np.cos(ang) + h * np.sin(num*ang) * np.cos(ang) #vector derivative
    dmag_inv = (dxdt**2 + dydt**2) ** -0.5 #inverse magnitude of derivative vector
    x_offset = dydt * dmag_inv * th
    y_offset = -dxdt * dmag_inv * th #vector offset from neutral axis position
    
    #Compute new point
    x_new = x_out + x_offset
    y_new = y_out + y_offset
    
    return x_new, y_new
    


#========================================POINT SAMPLING========================================


        
def surfingBC(e,t,Xc,cslist,vslist,E,nu,ampl,strainRate,slope):
    '''
    U = surfingBC(exoout,t,options.initialpos,options.cs,options.vs,options.E,options.nu,options.ampl,options.strainRate,options.slope)
    e = exoout, exodus file to output
    t = time
    Xc = initial position offset vector -> Xc[0]=X offset, Xc[1]=Y offset, Xc[2]=Z offset
    cslist = materials list for BC
    vslist = vertex list for BC
    E = youngs modulus 
    nu = poisson ratio
    ampl = displacement amplitude
    strain rate = rate/speed of strain
    slope = displacement slope
    '''
    dim = e.num_dimensions()
    X,Y,Z=e.get_coords()
    U = np.zeros([3,len(X)]) #create empty vector
    
    csoffset = [e.elem_blk_info(set)[1] for set in cslist] #start list at 1
    
    for set in cslist:
        connect = e.get_elem_connectivity(set)
        for cid in range(connect[1]):
            vertices = [connect[0][cid*connect[2]+c] for c in range(connect[2])]
            for v in vertices:
                                
                #CS set list
                i = v-1 #actual point index
                
                #Calculate position of point after displacement
                output_closest_na = na1_pt_data(x1_vals, y1_vals, X[i], Y[i]) #compute thickness (th), arclength (s), and angle (theta) of point
                new_pt = mapped_pt(R2, h2, n, output_closest_na[0], output_closest_na[1], x2_vals, y2_vals) #compute new point using s and th
                       
                #Find angle to rotate vector
                theta = output_closest_na[2] #angle of original point
                bmps = np.floor( theta / bmp_ang ) #number of bumps to rotate
                rot_ang = bmp_ang * bmps
                       
                #Get displacement vector for point
                rotated_pt = rotate(X[i], Y[i], -rot_ang) #move to first section
                temp_dx = (new_pt[0] - rotated_pt[0]) * t * strainRate
                temp_dy = (new_pt[1] - rotated_pt[1]) * t * strainRate #total displacement * amount displaced at time              
                dspl_vec = rotate(temp_dx, temp_dy, rot_ang) #rotate to org position, clockwise
                
                #Set U value -> structured as [ {X,Y,Z}, {number of node} ]
                U[0,i] = dspl_vec[0]
                U[1,i] = dspl_vec[1]
                U[2,i] = 0 #Z-coord should be constant and 0
       
    for set in vslist:
        for v in e.get_node_set_nodes(set):
            
            #VS set list. Copy from above
            i = v-1 #actual point index
            
            #Calculate position of point after displacement
            output_closest_na = na1_pt_data(x1_vals, y1_vals, X[i], Y[i]) #compute thickness (th), arclength (s), and angle (theta) of point
            new_pt = mapped_pt(R2, h2, n, output_closest_na[0], output_closest_na[1], x2_vals, y2_vals) #compute new point using s and th
                   
            #Find angle to rotate vector
            theta = output_closest_na[2] #angle of original point
            bmps = np.floor( theta / bmp_ang ) #number of bumps to rotate
            rot_ang = bmp_ang * bmps
                   
            #Get displacement vector for point
            rotated_pt = rotate(X[i], Y[i], -rot_ang) #move to first section
            temp_dx = (new_pt[0] - rotated_pt[0]) * t * strainRate
            temp_dy = (new_pt[1] - rotated_pt[1]) * t * strainRate #total displacement * amount displaced at time              
            dspl_vec = rotate(temp_dx, temp_dy, rot_ang) #rotate to org position, clockwise
            
            #Set U value -> structured as [ {X,Y,Z}, {number of node} ]
            U[0,i] = dspl_vec[0]
            U[1,i] = dspl_vec[1]
            U[2,i] = 0 #Z-coord should be constant and 0    
 
    return U



#========================================SYSTEM OPERATION========================================



def main():
    import numpy as np
    import os
    #print('hello')
    options = parse()
    #print('hello')
    if  os.path.exists(options.outputfile):
        if options.force:
            os.remove(options.outputfile)
        else:
            if pymef90.confirm("ExodusII file {0} already exists. Overwrite?".format(options.outputfile)):
                os.remove(options.outputfile)
            else:
                print('\n\t{0} was NOT generated.\n'.format(options.outputfile))
                return -1
    exoin  = exo.exodus(options.inputfile,mode='r')
    exoout = exoin.copy(options.outputfile)
    exoout.close()
    exoout  = exo.exodus(options.outputfile,mode='a',array_type='numpy')
    ### Adding a QA record, needed until visit fixes its exodus reader
    import datetime
    import os.path
    import sys
    QA_rec_len = 32
    QA = [os.path.basename(sys.argv[0]),os.path.basename(__file__),datetime.date.today().strftime('%Y%m%d'),datetime.datetime.now().strftime("%H:%M:%S")]
    exoout.put_qa_records([[ q[0:31] for q in QA],])

    exoformat(exoout,options.plasticity)
    
    dim = exoout.num_dimensions()
    step = 0
    frame_start_time = time.time() #start timer
    print(" ")
    print(">PROGRESS: Computing displacement...")
    for t in np.linspace(options.time_min,options.time_max,options.time_numstep):
        print("writing step {0}, t = {1:0.4f}".format(step+1,t)+"s")
        exoout.put_time(step+1,t)
        U = surfingBC(exoout,t,options.initialpos,options.cs,options.vs,options.E,options.nu,options.ampl,options.strainRate,options.slope)
        X,Y,Z=exoout.get_coords()
        exoout.put_node_variable_values("Displacement_X",step+1,U[0,:])
        exoout.put_node_variable_values("Displacement_Y",step+1,U[1,:])
        if dim == 3:
            exoout.put_node_variable_values("Displacement_Z",step+1,U[2,:])
        step += 1
        #calculate elapsed time for each frame
        frame_end_time = time.time()
        frame_exe_time = frame_end_time - frame_start_time
        print("-- elapsed time = {1:0.4f}".format(step+1,frame_exe_time)+"s")
    print(">PROGRESS: Saving file...")
    exoout.close()
    print(">PROGRESS: Done.")
    ### 
    ### compute boundary displacement at vertex sets
    ###
       
  
    
if __name__ == "__main__":
    import sys
    sys.exit(main())
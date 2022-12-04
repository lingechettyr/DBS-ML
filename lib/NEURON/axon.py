'''
    This file contains a python class, 'Axon', that allows the user to 
    create and stimulate multicompartment models of myelinated axons
    using the McIntyre Richardson Grill (MRG) model.
'''

import math
import numpy as np
from neuron import h

FIBERGEN_MSG = 0

class Axon:
    h.celsius = 37
    h.v_init = -80.0 # mV
    h.dt = 0.005 # Neuron parameter of time increments for simulation in ms 
    h('STINmul = 10') # Hoc variable for num of STIN sections
    STINmul = 10 # Python variable for num of STIN sections
    h.STINmul = STINmul
    compartDiv = STINmul + 5 # Number of sections in a compartment (1 node + 2 MYSA + 2 FLUT + 10 STIN)   
    
    pre_delay = 1
    post_delay = 2

    def __init__(self, diameter, number_of_nodes):

        # Topological parameters
        h('axonnodes = 0')
        h.axonnodes = number_of_nodes
        h('numsections = 0')
        h.numsections = self.sectionCount()
        h('double position[numsections]')
        h('create allsections[numsections]')
        h('access allsections[0]')
        diam = diameter # Diameter for each fiber in model space

        # Initialize vector arrays
        h('numvectors = 0') # Initialize variable for number of vectors (sections) in fiber
        h.numvectors = self.sectionCount() # Number of recording vectors (= total number of sections )
        h('objref vectors[numvectors]') # Initialize variable with length of numvectors
        h('objref stim_time, recordingvec')
        global voltages1 # Initialize voltages list for electrode 1

        h('stimsize = 0') # Stimulation amplitude initialization
        h('stim_time = new Vector()') # Initializes timing vector of stimulation

        # Neuron vectors for each waveform
        h('objref stim_time1')
        h('stimsize1 = 0')
        h('stim_time1 = new Vector()')

        # Topological parameters of fiber model (created in Neuron)
        h('fiberD=5.7') # changed from 4
        h('paralength1=3')  # Neuron parameter initialization of length of MYSA compartments in um
        h('nodelength=1.0')  # Neuron parameter initialization of length of nodes in um
        h('space_p1=0.002')
        h('space_p2=0.004')
        h('space_i=0.004')

        # Morphological Parameters
        h.fiberD = diam
        paralength1 = h.paralength1
        nodelength = h.nodelength

        # Electrical Parameters
        h('rhoa=700000.0') # Neuron parameter initialization of cytoplasmic resistance in ohm-um
        h('mycm=0.1') # Neuron parameter initialization of membrane capacitance in uF/cm^2 
        h('mygm=0.001') # Neuron parameter initialization of membrane conductance in S/cm^2
        rhoa = h.rhoa

        # FIXME The below is Brook's code. Need to verify that all diameters work in the future FIXME
        # proc dependent_var, have changed to 6 denom of interlength
        h("""
        if (fiberD==5.7) {g=0.605 axonD=3.4 nodeD=1.9 paraD1=1.9 paraD2=3.4 deltax=500 paralength2=35 nl=80}
        if (fiberD==2.0) {g=0.52 axonD=2.366 nodeD=0.7 paraD1=0.7 paraD2=1.448 deltax=200 paralength2=24.488 nl=50}
        if (fiberD==2.25) {g=0.52 axonD=2.44 nodeD=0.775 paraD1=0.775 paraD2=1.564 deltax=225 paralength2=25.137 nl=52}
        if (fiberD==2.5) {g=0.52 axonD=2.517 nodeD=0.85 paraD1=0.85 paraD2=1.684 deltax=250 paralength2=25.785 nl=55}
        if (fiberD==2.75) {g=0.52 axonD=2.596 nodeD=0.925 paraD1=0.925 paraD2=1.806 deltax=275 paralength2=26.434 nl=57}
        if (fiberD==3.0) {g=0.52 axonD=2.677 nodeD=1.0 paraD1=1.0 paraD2=1.931 deltax=300 paralength2=27.082 nl=60}
        if (fiberD==3.25) {g=0.52 axonD=2.762 nodeD=1.08 paraD1=1.08 paraD2=2.059 deltax=325 paralength2=27.731 nl=62}
        if (fiberD==3.5) {g=0.52 axonD=2.848 nodeD=1.16 paraD1=1.16 paraD2=2.19 deltax=350 paralength2=28.379 nl=65}
        if (fiberD==3.75) {g=0.52 axonD=2.938 nodeD=1.24 paraD1=1.24 paraD2=2.325 deltax=375 paralength2=29.028 nl=67}
        if (fiberD==4.0) {g=0.52 axonD=3.03 nodeD=1.33 paraD1=1.33 paraD2=2.462 deltax=400 paralength2=29.676 nl=70}
        if (fiberD==4.25) {g=0.52 axonD=3.125 nodeD=1.41 paraD1=1.41 paraD2=2.602 deltax=425 paralength2=30.325 nl=72}
        if (fiberD==4.5) {g=0.52 axonD=3.223 nodeD=1.49 paraD1=1.49 paraD2=2.746 deltax=450 paralength2=30.973 nl=75}
        if (fiberD==4.75) {g=0.52 axonD=3.325 nodeD=1.58 paraD1=1.58 paraD2=2.892 deltax=475 paralength2=31.622 nl=77}
        if (fiberD==5.0) {g=0.52 axonD=3.429 nodeD=1.67 paraD1=1.67 paraD2=3.042 deltax=500 paralength2=32.271 nl=80}
        if (fiberD==5.25) {g=0.52 axonD=3.537 nodeD=1.75 paraD1=1.75 paraD2=3.194 deltax=525 paralength2=32.919 nl=82}
        if (fiberD==5.5) {g=0.52 axonD=3.648 nodeD=1.83 paraD1=1.83 paraD2=3.35 deltax=550 paralength2=33.568 nl=85}
        if (fiberD==5.75) {g=0.52 axonD=3.762 nodeD=1.91 paraD1=1.91 paraD2=3.508 deltax=575 paralength2=34.216 nl=87}
        if (fiberD==6.0) {g=0.52 axonD=3.881 nodeD=2.0 paraD1=2.0 paraD2=3.67 deltax=600 paralength2=34.865 nl=90}
        if (fiberD==6.25) {g=0.52 axonD=4.002 nodeD=2.08 paraD1=2.08 paraD2=3.835 deltax=625 paralength2=35.513 nl=92}
        if (fiberD==6.5) {g=0.52 axonD=4.128 nodeD=2.16 paraD1=2.16 paraD2=4.002 deltax=650 paralength2=36.162 nl=95}
        if (fiberD==6.75) {g=0.52 axonD=4.258 nodeD=2.24 paraD1=2.24 paraD2=4.173 deltax=675 paralength2=36.81 nl=97}
        if (fiberD==7.0) {g=0.52 axonD=4.392 nodeD=2.33 paraD1=2.33 paraD2=4.347 deltax=700 paralength2=37.459 nl=100}
        if (fiberD==7.25) {g=0.52 axonD=4.529 nodeD=2.41 paraD1=2.41 paraD2=4.524 deltax=725 paralength2=38.107 nl=102}
        if (fiberD==7.5) {g=0.52 axonD=4.672 nodeD=2.49 paraD1=2.49 paraD2=4.704 deltax=750 paralength2=38.756 nl=105}
        if (fiberD==7.75) {g=0.52 axonD=4.818 nodeD=2.58 paraD1=2.58 paraD2=4.886 deltax=775 paralength2=39.404 nl=107}
        if (fiberD==8.0) {g=0.52 axonD=4.97 nodeD=2.67 paraD1=2.67 paraD2=5.072 deltax=800 paralength2=40.053 nl=110}
        if (fiberD==8.25) {g=0.52 axonD=5.126 nodeD=2.75 paraD1=2.75 paraD2=5.261 deltax=825 paralength2=40.701 nl=111}
        if (fiberD==8.5) {g=0.52 axonD=5.287 nodeD=2.83 paraD1=2.83 paraD2=5.453 deltax=850 paralength2=41.35 nl=112}
        if (fiberD==8.75) {g=0.52 axonD=5.453 nodeD=2.91 paraD1=2.91 paraD2=5.648 deltax=875 paralength2=41.998 nl=114}
        if (fiberD==9.0) {g=0.52 axonD=5.624 nodeD=3.0 paraD1=3.0 paraD2=5.847 deltax=900 paralength2=42.647 nl=115}
        if (fiberD==9.25) {g=0.52 axonD=5.801 nodeD=3.08 paraD1=3.08 paraD2=6.048 deltax=925 paralength2=43.295 nl=116}
        if (fiberD==9.5) {g=0.52 axonD=5.983 nodeD=3.16 paraD1=3.16 paraD2=6.252 deltax=950 paralength2=43.944 nl=117}
        if (fiberD==9.75) {g=0.52 axonD=6.171 nodeD=3.24 paraD1=3.24 paraD2=6.459 deltax=975 paralength2=44.592 nl=119}
        if (fiberD==10.0) {g=0.52 axonD=6.365 nodeD=3.33 paraD1=3.33 paraD2=6.669 deltax=1000 paralength2=45.241 nl=120}
        if (fiberD==10.25) {g=0.52 axonD=6.565 nodeD=3.41 paraD1=3.41 paraD2=6.882 deltax=1025 paralength2=45.89 nl=121}
        if (fiberD==10.5) {g=0.52 axonD=6.771 nodeD=3.49 paraD1=3.49 paraD2=7.099 deltax=1050 paralength2=46.538 nl=122}
        if (fiberD==10.75) {g=0.52 axonD=6.984 nodeD=3.58 paraD1=3.58 paraD2=7.318 deltax=1075 paralength2=47.187 nl=124}
        if (fiberD==11.0) {g=0.52 axonD=7.203 nodeD=3.67 paraD1=3.67 paraD2=7.541 deltax=1100 paralength2=47.835 nl=125}
        if (fiberD==11.25) {g=0.52 axonD=7.429 nodeD=3.75 paraD1=3.75 paraD2=7.766 deltax=1125 paralength2=48.484 nl=126}
        if (fiberD==11.5) {g=0.52 axonD=7.662 nodeD=3.83 paraD1=3.83 paraD2=7.994 deltax=1150 paralength2=49.132 nl=127}
        if (fiberD==11.75) {g=0.52 axonD=7.903 nodeD=3.91 paraD1=3.91 paraD2=8.226 deltax=1175 paralength2=49.781 nl=129}
        if (fiberD==12.0) {g=0.52 axonD=8.151 nodeD=4.0 paraD1=4.0 paraD2=8.46 deltax=1200 paralength2=50.429 nl=130}
        if (fiberD==12.25) {g=0.52 axonD=8.407 nodeD=4.08 paraD1=4.08 paraD2=8.698 deltax=1225 paralength2=51.078 nl=131}
        if (fiberD==12.5) {g=0.52 axonD=8.671 nodeD=4.16 paraD1=4.16 paraD2=8.939 deltax=1250 paralength2=51.726 nl=132}
        if (fiberD==12.75) {g=0.52 axonD=8.944 nodeD=4.24 paraD1=4.24 paraD2=9.182 deltax=1275 paralength2=52.375 nl=133}
        if (fiberD==13.0) {g=0.52 axonD=9.225 nodeD=4.33 paraD1=4.33 paraD2=9.429 deltax=1300 paralength2=53.023 nl=135}
        if (fiberD==13.25) {g=0.52 axonD=9.514 nodeD=4.41 paraD1=4.41 paraD2=9.679 deltax=1325 paralength2=53.672 nl=136}
        if (fiberD==13.5) {g=0.52 axonD=9.813 nodeD=4.49 paraD1=4.49 paraD2=9.931 deltax=1350 paralength2=54.32 nl=137}
        if (fiberD==13.75) {g=0.52 axonD=10.121 nodeD=4.58 paraD1=4.58 paraD2=10.187 deltax=1375 paralength2=54.969 nl=139}
        if (fiberD==14.0) {g=0.52 axonD=10.439 nodeD=4.67 paraD1=4.67 paraD2=10.446 deltax=1400 paralength2=55.617 nl=140}
        if (fiberD==14.25) {g=0.52 axonD=10.767 nodeD=4.75 paraD1=4.75 paraD2=10.708 deltax=1425 paralength2=56.266 nl=141}
        if (fiberD==14.5) {g=0.52 axonD=11.105 nodeD=4.83 paraD1=4.83 paraD2=10.973 deltax=1450 paralength2=56.914 nl=142}
        if (fiberD==14.75) {g=0.52 axonD=11.454 nodeD=4.91 paraD1=4.91 paraD2=11.241 deltax=1475 paralength2=57.563 nl=145}
        if (fiberD==15.0) {g=0.52 axonD=11.814 nodeD=5.0 paraD1=5.0 paraD2=11.512 deltax=1500 paralength2=58.212 nl=145}
        if (fiberD==15.25) {g=0.52 axonD=12.185 nodeD=5.08 paraD1=5.08 paraD2=11.786 deltax=1525 paralength2=58.86 nl=146}
        if (fiberD==15.5) {g=0.52 axonD=12.568 nodeD=5.16 paraD1=5.16 paraD2=12.063 deltax=1550 paralength2=59.509 nl=147}
        if (fiberD==15.75) {g=0.52 axonD=12.962 nodeD=5.24 paraD1=5.24 paraD2=12.343 deltax=1575 paralength2=60.157 nl=149}
        if (fiberD==16.0) {g=0.52 axonD=13.37 nodeD=5.33 paraD1=5.33 paraD2=12.626 deltax=1600 paralength2=60.806 nl=150}

        Rpn0=(rhoa*.01)/(PI*((((nodeD/2)+space_p1)^2)-((nodeD/2)^2)))
        Rpn1=(rhoa*.01)/(PI*((((paraD1/2)+space_p1)^2)-((paraD1/2)^2)))
        Rpn2=(rhoa*.01)/(PI*((((paraD2/2)+space_p2)^2)-((paraD2/2)^2)))
        Rpx=(rhoa*.01)/(PI*((((axonD/2)+space_i)^2)-((axonD/2)^2)))
        interlength=(deltax-nodelength-(2*paralength1)-(2*paralength2))/STINmul
        Gan=(1/(rhoa*paralength1/(PI*(paraD1/2)^2)))
        Gax1=(1/(rhoa*4/(PI*(paraD2/2)^2)))
        Gax2=(1/(rhoa*interlength/(PI*(paraD2/2)^2)))
            """)

        h('curpos = 0')
        h('curIndex = 0')
        
        for index in range(self.sectionCount()):   
            h.curIndex = index

            # Create an active node at every node not at either end
            if index % self.compartDiv == 0 and index != 0 and index != ((h.axonnodes-1)*self.compartDiv):
                h.allsections[index].nseg = 1
                h.allsections[index].diam = h.nodeD
                h.allsections[index].L = nodelength
                h.allsections[index].Ra = rhoa/10000
                h.allsections[index].cm = 2
                h.allsections[index].insert('axnode')
                h('allsections[curIndex] insert extracellular')
                h('allsections[curIndex] xraxial=Rpn0')
                h('allsections[curIndex] xg=1e10')
                h('allsections[curIndex] xc=0')
                if FIBERGEN_MSG:
                    print ("creating node at index: ", index)
            
            # Create a passive node at the end nodes
            if index == 0 or index == ((h.axonnodes-1)*self.compartDiv):
                h.allsections[index].nseg = 1
                h.allsections[index].diam = h.nodeD
                h.allsections[index].L = nodelength
                h.allsections[index].Ra = rhoa/10000
                h.allsections[index].cm = 2
                h.allsections[index].insert('pas')
                h('allsections[curIndex] insert extracellular')
                h('allsections[curIndex] xraxial=Rpn0')
                h('allsections[curIndex] xg=1e10')
                h('allsections[curIndex] xc=0')
                if FIBERGEN_MSG:
                    print ("creating passive node at index: ", index)
        
            # Create MYSA compartments at the adjoining compartments to the nodes 
            if index % self.compartDiv == 1 or index % self.compartDiv == (self.compartDiv-1): # is MYSA 
                h.allsections[index].nseg = 1
                h.allsections[index].diam = h.fiberD
                h.allsections[index].L = paralength1
                h.allsections[index].Ra = rhoa * (1/math.pow(h.paraD1/h.fiberD, 2.0))/10000
                h.allsections[index].cm = 2 * h.paraD1/h.fiberD
                h.allsections[index].insert('pas')
                h.allsections[index].g_pas = 0.001 * h.paraD1/h.fiberD
                h.allsections[index].e_pas = -80
                h('allsections[curIndex] insert extracellular')
                h('allsections[curIndex] xraxial=Rpn1')
                h('allsections[curIndex] xg=mygm/(nl*2)')
                h('allsections[curIndex] xc=mycm/(nl*2)')
                if FIBERGEN_MSG:
                    print ("creating MYSA at index: ", index)

            # Create FLUT compartments in between MYSA and STIN compartments      
            if index % self.compartDiv == 2 or index % self.compartDiv == (self.compartDiv-2): # is FLUT
                h.allsections[index].nseg = 1
                h.allsections[index].diam = h.fiberD
                h.allsections[index].L = h.paralength2
                h.allsections[index].Ra = rhoa * (1/math.pow(h.paraD2/h.fiberD, 2.0))/10000
                h.allsections[index].cm = 2 * h.paraD2/h.fiberD
                h.allsections[index].insert('pas')
                h.allsections[index].g_pas = 0.0001 * h.paraD2/h.fiberD
                h.allsections[index].e_pas = -80
                h('allsections[curIndex] insert extracellular')
                h('allsections[curIndex] xraxial=Rpn2')
                h('allsections[curIndex] xg=mygm/(nl*2)')
                h('allsections[curIndex] xc=mycm/(nl*2)')
                if FIBERGEN_MSG:
                    print ("creating FLUT at index: ", index)

            # Create STIN compartments in between FLUT compartments
            if index % self.compartDiv > 2 and index % self.compartDiv < (self.compartDiv-2): # is STIN
                h.allsections[index].nseg = 1
                h.allsections[index].diam = h.fiberD
                h.allsections[index].L = h.interlength
                h.allsections[index].Ra = rhoa * (1/math.pow(h.axonD/h.fiberD, 2.0))/10000
                h.allsections[index].cm = 2 * h.axonD/h.fiberD
                h.allsections[index].insert('pas')
                h.allsections[index].g_pas = 0.0001 * h.axonD/h.fiberD
                h.allsections[index].e_pas = -80
                h('allsections[curIndex] insert extracellular')
                h('allsections[curIndex] xraxial=Rpx')
                h('allsections[curIndex] xg=mygm/(nl*2)')
                h('allsections[curIndex] xc=mycm/(nl*2)')
                if FIBERGEN_MSG:
                    print ("creating STIN at index: ", index)

            # If compartment is not the first (index == 0) then connect it to the previous compartment
            if index > 0: 
                h('connect allsections[curIndex](0), allsections[curIndex - 1](1)')
                if FIBERGEN_MSG:
                    print ("connecting ", index, " to ", index - 1)

            # Use the length variables to create an array of positions
            h('curpos = curpos + (allsections[curIndex].L/2)')
            h('position[curIndex] = curpos')
            h('curpos = curpos + (allsections[curIndex].L/2)')

        h('finitialize(v_init)')
        h('fcurrent()')

    def stimulate(self, voltages, multiplier, pulse_width, frequency, number_of_pulses):
        h('objref vectors1[numvectors]')
        wf_period = ((1000/(1*frequency)))
        h.stimsize1=4*number_of_pulses+2  # Calculates how big stimulation vector will need to be based off of pulses desired
        h.stim_time1.resize(h.stimsize1)
        for index in range(int(len(voltages))):
            h.curIndex = index
            h('vectors1[curIndex] = new Vector(stimsize1, 0)')

        h.tstop = self.pre_delay + wf_period*(number_of_pulses-1) + self.post_delay

        for i in range(number_of_pulses):
            h.stim_time1.x[i*4+1]=self.pre_delay+i*(wf_period)
            h.stim_time1.x[i*4+2]=self.pre_delay+i*(wf_period)
            h.stim_time1.x[i*4+3]=self.pre_delay+pulse_width+i*(wf_period)
            h.stim_time1.x[i*4+4]=self.pre_delay+pulse_width+i*(wf_period)
            for index in range(int(len(voltages))):
                h.vectors1[index].x[i*4+2]=-multiplier*voltages[index] #FIXME
                h.vectors1[index].x[i*4+3]=-multiplier*voltages[index] #FIXME
        
        h.stim_time1.x[int(h.stimsize1-1)] = h.tstop

        # Iterate list of vects to play them  
        for index in range(0,int(len(voltages)),1):
            h.curIndex = index         
            h.vectors1[index].play(h.allsections[index](0.5)._ref_e_extracellular, h.stim_time1, 1)

        # Set up the APCount objects to later detect spikes
        apc = []
        for i in range(int(h.axonnodes)):
            node_ind = i * self.compartDiv
            apc_temp = h.APCount(h.allsections[node_ind](0.5))
            apc_temp.n = 0
            apc_temp.thresh = 0
            apc.append(apc_temp)

        h.init()
        h.run()

        # loop through the APCount objects to determine whether a spike propagated, and from which node
        spike_count = 0
        ap_prop_count = 0
        first_spike_time = 100000
        first_spike_ind = -1
        for i in range(len(apc)):
            temp_spike_time = apc[i].time
            if apc[i].n >= 1:
                ap_prop_count += 1
                if temp_spike_time < first_spike_time:
                    first_spike_time = temp_spike_time
                    first_spike_ind = i
        
        if ap_prop_count > 10:
            spike_count = apc[first_spike_ind].n

        if spike_count > 0:
            first_spike_inds = []
            for i in range(len(apc)):
                temp_spike_time = apc[i].time
                if apc[i].n >= 1 and temp_spike_time == first_spike_time:
                    first_spike_inds.append(i)
        else:
            first_spike_inds = []
                
        return spike_count, first_spike_inds

    def findThreshold(self, voltages, pulse_width, frequency, number_of_pulses):
        expected_spikes = number_of_pulses # Define expected number of spikes (should be number of pulses or waveforms) 
        precision = 0.001 # specific the desired threshold precision
        upper_stim_bound = 11 #FIXME change back to 50 V

        # Utilizes an exponential search
        # Start with a ladder search with doubling steps to identify bounds for the binary search (needed to avoid overstimulation)
        multiplier = 0.05
        multiplier_prev = 0
        while(multiplier_prev < upper_stim_bound):
            spike_count, _ = self.stimulate(voltages, multiplier, pulse_width, frequency, number_of_pulses)
            print("Ladder search: multiplier = " + str(multiplier) + ", spikes = " + str(spike_count))
            if spike_count >= expected_spikes:
                upper_bound = multiplier
                lower_bound = multiplier_prev
                break
            else:
                multiplier_prev = multiplier
                multiplier *= 2
        
        if multiplier_prev >= upper_stim_bound:
            print("Threshold not less than upper stim bound: possible overstimulation")
            return upper_stim_bound

        print()

        # Do a binary search on the bounds from the prior ladder search
        iterations = math.ceil(math.log((upper_bound-lower_bound)/precision,2)) #calculate the number of binary search iterations to achieve a certain precision
        for i in range(iterations):
            multiplier = ((upper_bound + lower_bound) / 2)
            spike_count, _ = self.stimulate(voltages, multiplier, pulse_width, frequency, number_of_pulses)
            print("Binary search: iteration = " + str(i) + ", multiplier = " + str(multiplier) + ", spikes = " + str(spike_count))
            if spike_count >= expected_spikes:
                upper_bound = multiplier
            else:
                lower_bound = multiplier

        multiplier = ((upper_bound + lower_bound) / 2)
        roundTo = 2 + int(abs(math.log(precision, 10)))
        threshold = round(multiplier, roundTo)

        print()

        return threshold

    def sectionCount(self):
        compartmentCount = int(h.axonnodes - 1)
        totalMYSA = 2 * compartmentCount # myelin attachment segments
        totalFLUT = 2 * compartmentCount # paranodal main segments
        totalSTIN = self.STINmul * compartmentCount # 10 internodal main segments
        return int(h.axonnodes) + totalMYSA + totalFLUT + totalSTIN

    def getCompartmentPositions(self):
        return h.position

    def getDeltax(self):
        return h.deltax
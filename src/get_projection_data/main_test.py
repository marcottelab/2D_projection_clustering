import numpy as np
import mrcfile
from slicem import get_projection_2D
import random


# takes list of pdbs and converts to projected mrcs
def makeImages(path, pdbinput,n_projs=30,moleculename=''):
    from biopandas.pdb import PandasPdb
    import subprocess
    import os
    #ppdb = PandasPdb().fetch_pdb(pdbinput)
    #COMPNDdf = ppdb.df['OTHERS'].loc[ppdb.df['OTHERS']['record_name']  == 'COMPND']
    #moleculeline = COMPNDdf[COMPNDdf['entry'].str.contains(" MOLECULE:")]['entry'].iloc[0]
    #moleculename = moleculeline[moleculeline.find(':')+1 : moleculeline.find(';')].replace(' ', '')
    #moleculename =''.join(e for e in moleculename if e.isalnum())
    #ppdb.to_pdb(path=path + pdbinput + '.pdb')
    #print(moleculename)

    # lowpass filter pdb to resolution 3
    
    # subprocess.run([r"/mnt/c/EMAN/bin/pdb2mrc", path + pdbinput + '.pdb', path + moleculename + '_' + pdbinput + '.mrc', 'apix=1', 'res=9', 'center'])
    subprocess.run([r"/home/meghana/relion/build/bin/relion_project", '--i', path + moleculename + '_' + pdbinput + '.mrc', '--o', path + moleculename + '_' + pdbinput + '_proj', '--nr_uniform', str(n_projs)])
    
    
    #subprocess.run([r"C:\EMAN\bin\pdb2mrc", path + pdbinput + '.pdb', path + moleculename + '_' + pdbinput + '.mrc', 'apix=1', 'res=9', 'center'])
    #subprocess.run([r"C:\Users\Meghana\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\meghana\relion\build\bin\relion_project", '--i', path + moleculename + '_' + pdbinput + '.mrc', '--o', path + moleculename + '_' + pdbinput + '_proj', '--nr_uniform', '30'])
    # #subprocess.run([r"./relion/build/bin/relion_project", '--i', path + moleculename + '_' + pdbinput + '.mrc', '--o', path + moleculename + '_' + pdbinput + '_proj', '--nr_uniform', '30'])
    

    # delete pdb and mrc files
    # os.remove(path + pdbinput + '.pdb')
    # try:
    #     os.remove(path + moleculename + '_' + pdbinput + '.mrc')
    # except FileNotFoundError:
    #     pass
    
    # try:
    #     os.remove(path + moleculename + '_' + pdbinput + '_proj.star')
    # except FileNotFoundError:
    #     pass

    mrcname = moleculename + '_' + pdbinput + '_proj.mrcs'
    return moleculename, mrcname

out_path = 'C:\\Users\\Meghana\\Box Sync\\Research\\EdwardMarcotte\\CryoEMproject\\code_data_results\\data\\synthetic_more_projections\\'
#out_path = r"/mnt/c/Users/Meghana/Box Sync/Research/EdwardMarcotte/CryoEMproject/code_data_results/data/synthetic_more_projections/"


pdb_names = ['1A0I', '1HHO', '1NW9', '1WA5', '3JCK', '5A63', '1A36','1HNW', '1PJR', '2FFL', '3JCR', '5GJQ', '1AON', '1I6H', '1RYP', '2MYS', '3VKH','5VOX', '1FA0', '1JLB', '1S5L', '2NN6', '4F3T', '6B3R', '1FPY', '1MUH', '1SXJ','2SRC', '6D6V', '1GFL', '1NJI', '1TAU', '3JB9', '5A1A','4V6C']
#pdb_names = ['4V6C']

#pdb_names = ['1A0I']

mol_names = ['DNALIGASE',
'HEMOGLOBINAOXYALPHACHAIN',
'BACULOVIRALIAPREPEATCONTAININGPROTEIN4',
'GTPBINDINGNUCLEARPROTEINRAN',
'26SPROTEASOMEREGULATORYSUBUNITRPN3',
'NICASTRIN',
'DNA5',
'16SRIBOSOMALRNA',
'PCRA',
'DICER',
'HPRP6',
'PROTEASOMESUBUNITBETATYPE6',
'GROEL',
'5DPAPAPAPTPGPCPCPTPGPGPTPCPT3',
'20SPROTEASOME',
'MYOSIN',
'DYNEINHEAVYCHAINCYTOPLASMIC',
'VTYPEPROTONATPASECATALYTICSUBUNITAVTYPEPROTO',
'POLYAPOLYMERASE',
'HIV1RTACHAIN',
'PHOTOSYSTEMQBPROTEIN',
'POLYMYOSITISSCLERODERMAAUTOANTIGEN1',
'PROTEINARGONAUTE2',
'PIEZOTYPEMECHANOSENSITIVEIONCHANNELCOMPONENT1',
'GLUTAMINESYNTHETASE',
'DNATRANSFERREDSTRAND',
'ACTIVATOR195KDASUBUNIT',
'TYROSINEPROTEINKINASESRC',
'TELOMERASEREVERSETRANSCRIPTASE',
'GREENFLUORESCENTPROTEIN',
'23SRIBOSOMALRNA',
'DNA5DGPCPGPAPTPCPCPG3',
'PREMRNASPLICINGFACTORSPP42',
'BETAGALACTOSIDASE',
'70SRIBOSOME']

# mol_names = ['70SRIBOSOME']

# indices = []
# mrc_name_list = []

# import random
# i=0
# for ind,pdb in enumerate(pdb_names):
#     n_projs = random.randint(20,30)
#     indices.append(pdb+'\t'+'['+','.join([str(n) for n in list(range(i,i+n_projs))])+']'+'\n')
#     i = i + n_projs
#     if ind < len(mol_names):
#         mol_name, mrc_name = makeImages(out_path, pdb,n_projs,mol_names[ind])        
#     else:
#         mol_name, mrc_name = makeImages(out_path, pdb,n_projs)
#     mrc_name_list.append(mrc_name)
    
# with open(out_path+'true_clustering.txt','w') as f:
#     f.writelines(indices)
    


# mrc_name_list = [mol_names[i]+'_'+pdb_names[i]+'_proj.mrcs' for i in range(len(pdb_names))]

# mrc_name_list_new = []

# # # for i,pdb in enumerate(pdb_names):
# # #     mrc = mrcfile.open(out_path+mrc_name_list[i], mode='r+')
# # #     data = mrc.data
# # #     mrc_data_arr.append(data)
# # #     mrc.close()


# # Find max size
# max_size = 0
# new_sizes = []
# for i,pdb in enumerate(pdb_names):
#     mrc = mrcfile.open(out_path+mrc_name_list[i], mode='r+')
#     data = mrc.data
#     mrc.close()
    
#     n_projs = len(data)
    
#     n_projs_new = random.randint(13,20)
#     new_sizes.append(n_projs_new)
#     name_new = mrc_name_list[i] + '_' + str(n_projs_new) + '.mrcs'
#     mrc_name_list_new.append(name_new)
    
#     new_indices = random.sample(range(0, n_projs-1), n_projs_new)
    
#     data_new = data[new_indices]
    
#     # Downsample projections randomly
#     with mrcfile.new(out_path+name_new,overwrite=True) as mrc:
#           mrc.set_data(data_new)
          
#     max_size = max(max_size,np.shape(data_new[0])[0])
    
# with open(out_path+'name_list.txt','w') as f: 
#     f.writelines([name+'\n' for name in mrc_name_list_new])
    
# print(max_size)

# i=0
# for indexx, pdb in enumerate(pdb_names):
#     n_projs = new_sizes[indexx]
#     indices.append(pdb+'\t'+'['+','.join([str(n) for n in list(range(i,i+n_projs))])+']'+'\n')
#     i = i + n_projs
    
# with open(out_path+'true_clustering.txt','w') as f:
#     f.writelines(indices)

    
with open(out_path+'name_list.txt') as f: 
    mrc_name_list_new = [line.rstrip() for line in f.readlines()]
    
# Find max size
max_size = 0
new_sizes = []
for i,pdb in enumerate(pdb_names):
    mrc = mrcfile.open(out_path+mrc_name_list_new[i], mode='r+')
    data = mrc.data
    mrc.close()
          
    max_size = max(max_size,np.shape(data[0])[0])
    
print(max_size) #416

mrc_data_arr = []
for i,pdb in enumerate(pdb_names):
    shape, projection_2D = get_projection_2D(mrcs=out_path+mrc_name_list_new[i], factor=1,out_size=(100,100),resize=True,pad=max_size)
    projection_2D_arr = np.array(list(projection_2D.values()))
    mrc_data_arr.append(projection_2D_arr)
    
combined_mrc = np.concatenate(mrc_data_arr)

with mrcfile.new(out_path+'synthetic_more_projections.mrcs',overwrite=True) as mrc:
      mrc.set_data(combined_mrc)
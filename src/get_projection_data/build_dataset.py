from slicem import get_projection_2D
from biopandas.pdb import PandasPdb

import subprocess
import numpy as np
import mrcfile
import argparse
import random


def get_max_size(pdb_names,out_path,mrc_name_list_new):
    '''
    
    Find max size of molecules

    Parameters
    ----------
    pdb_names : list[str]
        List of pdb names
    out_path : str
        Name of path with pdb and mrc files
    mrc_name_list_new : list[str]
        List of mrc file names

    Returns
    -------
    max_size : int
        Dimension of the largest molecule mrc file

    '''
    max_size = 0
    for i,pdb in enumerate(pdb_names):
        mrc = mrcfile.open(out_path+mrc_name_list_new[i], mode='r+')
        data = mrc.data
        mrc.close()
              
        max_size = max(max_size,np.shape(data[0])[0])
        
    print(max_size) #416
    return max_size


def resize_concatenate_mrcs(pdb_names,out_path,mrc_name_list_new, max_size):
    '''
    Pad molecules to max size and concatenate and save to one file

    Parameters
    ----------
    pdb_names : list[str]
        List of pdb names
    out_path : str
        Name of path with pdb and mrc files
    mrc_name_list_new : list[str]
        List of mrc file names
    max_size : int
        Dimension of the largest molecule mrc file

    Returns
    -------
    None.

    '''
    mrc_data_arr = []
    for i,pdb in enumerate(pdb_names):
        shape, projection_2D = get_projection_2D(mrcs=out_path+mrc_name_list_new[i], factor=1,out_size=(100,100),resize=True,pad=max_size)
        projection_2D_arr = np.array(list(projection_2D.values()))
        mrc_data_arr.append(projection_2D_arr)
        
    combined_mrc = np.concatenate(mrc_data_arr)
    
    with mrcfile.new(out_path+'synthetic_more_projections.mrcs',overwrite=True) as mrc:
          mrc.set_data(combined_mrc)


def makeImages(path, pdbinput, n_projs=30, moleculename='', pdb2mrc_path=r"/mnt/c/EMAN/bin/pdb2mrc", relion_path = r"/home/meghana/relion/build/bin/relion_project",convert2mrc_flag = 1):
    """ 
    Get 2D projections and make mrc file for a moleucle. 

    Parameters
    ----------
    path : str
        Name of path to save pdb and mrc files
    pdbinput : str
        Name of pdb identifier for molecule
    n_projs : int, optional
        No. of 2D projections to uniformly project the molecule. The default is 30.
    moleculename : str, optional
        Name of the molecule. The default is ''.
    pdb2mrc_path : str, optional
        Path to pdb2mrc file, ex: r"C:\EMAN\bin\pdb2mrc". 
        The default is r"/mnt/c/EMAN/bin/pdb2mrc".
    relion_path : str, optional
        Path to relion path The default is r"/home/meghana/relion/build/bin/relion_project".

    Returns
    -------
    moleculename : str
        Name of the molecule
    mrcname : str
        Name of the mrc file  for the molecule

    """

    if moleculename == '' and (pdbinput != '4V6C'):
        ppdb = PandasPdb().fetch_pdb(pdbinput)
        COMPNDdf = ppdb.df['OTHERS'].loc[ppdb.df['OTHERS']['record_name']  == 'COMPND']
        moleculeline = COMPNDdf[COMPNDdf['entry'].str.contains(" MOLECULE:")]['entry'].iloc[0]
        moleculename = moleculeline[moleculeline.find(':')+1 : moleculeline.find(';')].replace(' ', '')
        moleculename =''.join(e for e in moleculename if e.isalnum())
        ppdb.to_pdb(path=path + pdbinput + '.pdb')
        print(moleculename)

    # lowpass filter pdb to resolution 3
    if convert2mrc_flag:
        subprocess.run([pdb2mrc_path, path + '/pdb/'+ pdbinput + '.pdb', path + '/mrc/'+ moleculename + '_' + pdbinput + '.mrc', 'apix=1', 'res=9', 'center'])
    
    subprocess.run([relion_path, '--i', path + '/mrc/' + moleculename + '_' + pdbinput + '.mrc', '--o', path + moleculename + '_' + pdbinput + '_proj', '--nr_uniform', str(n_projs)])
    
    mrcname = moleculename + '_' + pdbinput + '_proj.mrcs'
    return moleculename, mrcname


def get_and_write_mrcs_unif_projs(pdb_names, out_path, mol_names):
    '''
    Write mrcs of uniform projections of input molecules and the true clusters

    Parameters
    ----------
    pdb_names : list[str]
        List of pdb names
    out_path : str
        Name of path with pdb and mrc files
    mol_names : list[str]
        List of corresponding molecule names        

    Returns
    -------
    None.

    '''

    indices = []
    mrc_name_list = []  
    i=0
    for ind,pdb in enumerate(pdb_names):
        n_projs = random.randint(20,30)
        indices.append(pdb+'\t'+'['+','.join([str(n) for n in list(range(i,i+n_projs))])+']'+'\n')
        i = i + n_projs
        if ind < len(mol_names):
            mol_name, mrc_name = makeImages(out_path, pdb,n_projs,mol_names[ind])        
        else:
            mol_name, mrc_name = makeImages(out_path, pdb,n_projs)
        mrc_name_list.append(mrc_name)
        
    with open(out_path+'true_clustering.txt','w') as f:
        f.writelines(indices)
        

def sample_and_write_random_projs(pdb_names, out_path, mol_names):
    '''
    Downsample projections randomly and write mrcs, list of mrc names and true clusters

    Parameters
    ----------
    pdb_names : list[str]
        List of pdb names
    out_path : str
        Name of path with pdb and mrc files
    mol_names : list[str]
        List of corresponding molecule names 

    Returns
    -------
    None.

    '''

    mrc_name_list = [mol_names[i]+'_'+pdb_names[i]+'_proj.mrcs' for i in range(len(pdb_names))]
    
    mrc_name_list_new = []
    
    new_sizes = []
    for i,pdb in enumerate(pdb_names):
        mrc = mrcfile.open(out_path+mrc_name_list[i], mode='r+')
        data = mrc.data
        mrc.close()
        
        n_projs = len(data)
        
        n_projs_new = random.randint(13,20)
        new_sizes.append(n_projs_new)
        name_new = mrc_name_list[i] + '_' + str(n_projs_new) + '.mrcs'
        mrc_name_list_new.append(name_new)
        
        new_indices = random.sample(range(0, n_projs-1), n_projs_new)
        
        data_new = data[new_indices]
        
        # Downsample projections randomly
        with mrcfile.new(out_path+name_new,overwrite=True) as mrc:
              mrc.set_data(data_new)
              
        
    with open(out_path+'name_list.txt','w') as f: 
        f.writelines([name+'\n' for name in mrc_name_list_new])
    
    indices = []    
    i=0
    for indexx, pdb in enumerate(pdb_names):
        n_projs = new_sizes[indexx]
        indices.append(pdb+'\t'+'['+','.join([str(n) for n in list(range(i,i+n_projs))])+']'+'\n')
        i = i + n_projs
        
    with open(out_path+'true_clustering.txt','w') as f:
        f.writelines(indices)
        

def main():        
    parser = argparse.ArgumentParser(description='build dataset from pdb names')
    
    parser.add_argument('-o', '--out_path', action='store', dest='out_path', required=True,
                        default = 'C:\\Users\\Meghana\\Box Sync\\Research\\EdwardMarcotte\\CryoEMproject\\code_data_results\\data\\synthetic_more_projections\\',
                        help='path for output files, ex: r"/mnt/c/Users/Meghana/Box Sync/Research/EdwardMarcotte/CryoEMproject/code_data_results/data/synthetic_more_projections/"')
    
    parser.add_argument("--pdb_names", nargs='+', default=['1A0I', '1HHO', '1NW9', '1WA5', '3JCK', '5A63', '1A36','1HNW', '1PJR', '2FFL', '3JCR', '5GJQ', '1AON', '1I6H', '1RYP', '2MYS', '3VKH','5VOX', '1FA0', '1JLB', '1S5L', '2NN6', '4F3T', '6B3R', '1FPY', '1MUH', '1SXJ','2SRC', '6D6V', '1GFL', '1NJI', '1TAU', '3JB9', '5A1A','4V6C'],
                        help=" Name of pdb identifiers")
    
    parser.add_argument("--mol_names", nargs='+', default=['DNALIGASE','HEMOGLOBINAOXYALPHACHAIN','BACULOVIRALIAPREPEATCONTAININGPROTEIN4','GTPBINDINGNUCLEARPROTEINRAN','26SPROTEASOMEREGULATORYSUBUNITRPN3','NICASTRIN','DNA5','16SRIBOSOMALRNA','PCRA','DICER','HPRP6','PROTEASOMESUBUNITBETATYPE6','GROEL','5DPAPAPAPTPGPCPCPTPGPGPTPCPT3','20SPROTEASOME','MYOSIN','DYNEINHEAVYCHAINCYTOPLASMIC','VTYPEPROTONATPASECATALYTICSUBUNITAVTYPEPROTO','POLYAPOLYMERASE','HIV1RTACHAIN','PHOTOSYSTEMQBPROTEIN','POLYMYOSITISSCLERODERMAAUTOANTIGEN1','PROTEINARGONAUTE2','PIEZOTYPEMECHANOSENSITIVEIONCHANNELCOMPONENT1','GLUTAMINESYNTHETASE','DNATRANSFERREDSTRAND','ACTIVATOR195KDASUBUNIT','TYROSINEPROTEINKINASESRC','TELOMERASEREVERSETRANSCRIPTASE','GREENFLUORESCENTPROTEIN','23SRIBOSOMALRNA','DNA5DGPCPGPAPTPCPCPG3','PREMRNASPLICINGFACTORSPP42','BETAGALACTOSIDASE','70SRIBOSOME'],
                        help=" Name of molecules corresponding to the pdb identifiers")
    
    args = parser.parse_args()
    
    out_path = args.out_path
    pdb_names = args.pdb_names
    mol_names = args.mol_names
    
    get_and_write_mrcs_unif_projs(pdb_names, out_path, mol_names)
    
    sample_and_write_random_projs(pdb_names, out_path, mol_names)
        
    with open(out_path+'name_list.txt') as f: 
        mrc_name_list_new = [line.rstrip() for line in f.readlines()]
    
    max_size = get_max_size(pdb_names,out_path,mrc_name_list_new)
    
    resize_concatenate_mrcs(pdb_names,out_path,mrc_name_list_new, max_size)

if __name__ == "__main__":
    main()
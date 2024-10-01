import streamlit as st
import xmltodict
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import ReplaceCore, GetMolFrags, CombineMols
from itertools import product
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
import mols2grid
import streamlit.components.v1 as components
import molfil
from rdkit.Chem import Draw

def smi2cansmi(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        for atm in mol.GetAtoms():
            atm.SetIsotope(0)
        return Chem.MolToSmiles(mol)
    else:
        return None

def generate_replacement_dictionary(replacement_xml_file):
    ifs = open(replacement_xml_file)
    xml = ifs.read()
    d = xmltodict.parse(xml)

    replacement_dict = defaultdict(list)
    for k in d['R_replacements']['center']:
        fl = k['first_layer']
        if not isinstance(fl, list):
            fl = [fl]
        first_layer = [a['@SMILES'] for a in fl]
        sl = [x for x in [a.get('second_layer') for a in fl] if x]
        second_layer = []
        for row in sl:
            if not isinstance(row, list):
                row = [row]
            for r in row:
                second_layer.append(r['@SMILES'])
        replacement_dict[smi2cansmi(k['@SMILES'])] = [first_layer, second_layer]
    return replacement_dict

def get_connect_idx(mol):
    return max([x.GetIsotope() for x in mol.GetAtoms()])

def get_replacements(smi, replacement_dict):
    cansmi = smi2cansmi(smi)
    return replacement_dict.get(cansmi, [[cansmi], [cansmi]])

def prep_sidechain(smi):
    mol = Chem.MolFromSmiles(smi)
    rw_mol = Chem.RWMol(mol)
    remove_idx = -1
    for atm in rw_mol.GetAtoms():
        if atm.GetAtomicNum() == 0:
            remove_idx = atm.GetIdx()
            for nbr in atm.GetNeighbors():
                nbr.SetAtomMapNum(1)
    rw_mol.RemoveAtom(remove_idx)
    Chem.SanitizeMol(rw_mol)
    return rw_mol

def make_analogs(core, attach_pts, sidechains):
    prod_list = []
    for c in product(*sidechains):
        sidechain_mol_list = [prep_sidechain(x) for x in c]
        mol = Chem.RWMol(core)
        for start_atm, m in zip(attach_pts, sidechain_mol_list):
            mol = Chem.RWMol(CombineMols(mol, m))
            end_atm = -1
            for atm in mol.GetAtoms():
                if atm.GetAtomMapNum() == 1:
                    end_atm = atm.GetIdx()
            mol.AddBond(start_atm, end_atm, order=Chem.rdchem.BondType.SINGLE)
            for atm in mol.GetAtoms():
                atm.SetAtomMapNum(0)
        prod_list.append(Chem.MolToSmiles(mol))
    return prod_list

def main():
    st.title("R Group Replacements App - Knowdis")
    molecule_input = st.text_input("Enter a SMILES string:")

    if molecule_input:
        use_second_layer = True
        mol = Chem.MolFromSmiles(molecule_input)

        if mol is not None:
            mol_target = mol
            replacement_dict = generate_replacement_dictionary("top500_R_replacements.xml")
            st.subheader("2D Structure")
            st.image(Draw.MolToImage(mol, size=(300, 300)), use_column_width=False, width=300)
            selected_option = st.radio("Select an option:", ["By molecule core", "Use Murcko scaffold"])   
            if selected_option == "By molecule core":
                sub_input = st.text_input("Enter a molecule core SMARTS for replacement")
                if sub_input:
                  mol_core = Chem.MolFromSmarts(sub_input)
                  try:       
                     cond = mol_target.HasSubstructMatch(mol_core)
                     sidechain_mol = ReplaceCore(mol_target, mol_core, labelByIndex=True)
                     sidechain_frag_list = GetMolFrags(sidechain_mol, asMols=True)
                     attach_idx = [get_connect_idx(x) for x in sidechain_frag_list]
                     sidechain_smiles = [Chem.MolToSmiles(x) for x in sidechain_frag_list]
                     replacement_smiles = [get_replacements(x, replacement_dict) for x in sidechain_smiles]
                    
                     if use_second_layer:
                         tmp_smiles = [a + b for a, b in replacement_smiles]
                     else:
                         tmp_smiles = [x[0] for x in replacement_smiles]
                     replacement_smiles = tmp_smiles
                     clean_sidechain_smiles = [smi2cansmi(x) for x in sidechain_smiles]
                     tmp_list = []
                     for i, j in zip(clean_sidechain_smiles, replacement_smiles):
                          tmp_list.append([i] + j)
                     replacement_smiles = tmp_list
                     analog_list = make_analogs(mol_core, attach_idx, replacement_smiles)
                     unique_smiles=list(set([Chem.CanonSmiles(i) for i in analog_list]))
                     st.write("Molecules before applying filter")
                     generated_df = pd.DataFrame({"smiles": analog_list})
                     raw_html = mols2grid.display(generated_df, smiles_col='smiles')._repr_html_()
                     components.html(raw_html, width=900, height=900, scrolling=True)
                     csv_file = generated_df.to_csv(index=False)
                     st.download_button(
                         label="Download CSV",
                         data=csv_file,
                         file_name="generated_data.csv",
                         key='csv_download'
                     )
                     st.write("Molecules after applying filter")
                     unique_smiles=list(set([Chem.CanonSmiles(i) for i in analog_list]))
                     out=molfil.rule.filter(unique_smiles)
                     selected_smiles = []
                     for item in out:
                # Check if all properties except 'smiles' are False
                         if all(value is False for key, value in item.items() if key != 'smiles'):
                            selected_smiles.append(item['smiles'])
                     generated_df = pd.DataFrame({"smiles": selected_smiles})
                     raw_html = mols2grid.display(generated_df, smiles_col='smiles')._repr_html_()
                     components.html(raw_html, width=900, height=900, scrolling=True)
                     csv_file = generated_df.to_csv(index=False)
                     st.download_button(
                         label="Download CSV",
                         data=csv_file,
                         file_name="generated_data.csv",
                         key='csv_download'
                     )
                  except:
                   st.write("Not a proper substructure")
            elif selected_option == "Use Murcko scaffold":
                mol_core = MurckoScaffold.GetScaffoldForMol(mol)
                core_smarts = Chem.MolToSmarts(mol_core)
                core_smiles=Chem.MolToSmiles(mol_core)
                st.write("The SMILES pattern for the Murcko scaffold is : "+core_smiles)
                sidechain_mol = ReplaceCore(mol_target, mol_core, labelByIndex=True)
                sidechain_frag_list = GetMolFrags(sidechain_mol, asMols=True)
                attach_idx = [get_connect_idx(x) for x in sidechain_frag_list]
                sidechain_smiles = [Chem.MolToSmiles(x) for x in sidechain_frag_list]
                replacement_smiles = [get_replacements(x, replacement_dict) for x in sidechain_smiles]
                if use_second_layer:
                    tmp_smiles = [a + b for a, b in replacement_smiles]
                else:
                    tmp_smiles = [x[0] for x in replacement_smiles]
                replacement_smiles = tmp_smiles
                clean_sidechain_smiles = [smi2cansmi(x) for x in sidechain_smiles]
                tmp_list = []
                for i, j in zip(clean_sidechain_smiles, replacement_smiles):
                    tmp_list.append([i] + j)
                replacement_smiles = tmp_list 
                analog_list = make_analogs(mol_core, attach_idx, replacement_smiles)
                st.write("Molecules before applying filter")
                generated_df = pd.DataFrame({"smiles": analog_list})
                raw_html = mols2grid.display(generated_df, smiles_col='smiles')._repr_html_()
                components.html(raw_html, width=900, height=900, scrolling=True)
                csv_file = generated_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_file,
                    file_name="generated_data.csv",
                    key='csv_download'
                )
                st.write("Molecules after applying filter")
                unique_smiles=list(set([Chem.CanonSmiles(i) for i in analog_list]))
                out=molfil.rule.filter(unique_smiles)
                selected_smiles = []
                for item in out:
                # Check if all properties except 'smiles' are False
                    if all(value is False for key, value in item.items() if key != 'smiles'):
                       selected_smiles.append(item['smiles'])
                generated_df = pd.DataFrame({"smiles": selected_smiles})
                raw_html = mols2grid.display(generated_df, smiles_col='smiles')._repr_html_()
                components.html(raw_html, width=900, height=900, scrolling=True)
                csv_file = generated_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_file,
                    file_name="generated_data.csv",
                    key='csv_download'
                )
                
        else:
            st.write("Invalid Input Molecule")

# Run the app
if __name__ == "__main__":
    main()
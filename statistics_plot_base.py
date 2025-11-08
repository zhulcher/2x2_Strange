from save_plotly import *
import matplotlib.pyplot as plt
from analysis.analysis_cuts import *

plt.rcParams.update({
    'figure.figsize': (10, 6),            # Larger figure size
    'figure.dpi': 300,                    # High-resolution output
    'savefig.dpi': 300,                   # High-res for saved figures
    'axes.labelsize': 20,                 # Larger axis labels
    'axes.titlesize': 16,                 # Larger plot title size
    'xtick.labelsize': 20,                # Larger x-tick labels
    'ytick.labelsize': 20,                # Larger y-tick labels
    'legend.fontsize': 20,                # Larger legend font
    'lines.linewidth': 2,                 # Thicker plot lines
    'lines.markersize': 6,                # Larger marker sizes
    'grid.color': 'gray',                 # Soft gridline color
    'grid.alpha': 0.3,                    # Transparent gridlines
    'axes.grid': True,                    # Enable gridlines
    'font.family': 'serif',               # Use serif fonts (like in publications)
    # 'text.usetex': True,                  # Enable LaTeX rendering for text (if available)
})


import glob

def clear_html_files(folder_path):
    """
    Recursively remove all .html files in the given folder and its subfolders.
    
    Parameters:
        folder_path (str): Path to the root folder to clear .html files from.
    """
    # Recursively find all .html files in the folder and its subfolders
    html_files = glob.glob(os.path.join(folder_path, '**', '*.html'), recursive=True)
    print("clearing htmls in: ",folder_path)
    for file_path in html_files:
        try:
            os.remove(file_path)  # Delete the file
            # print(f"Deleted: {file_path}")
        except Exception as e:
            # print(f"Failed to delete {file_path}: {e}")
            pass

def create_html_filename(index, filepath,extra=""):
    """
    Create a new filename by appending the given index and changing the extension to `.html`.
    
    Parameters:
        index (int): The index to append.
        filepath (str): The original file path.
    
    Returns:
        str: The modified filename with the `.html` extension.
    """
    import os
    
    # Extract the file name without extension
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    
    # Create the new filename
    new_filename = f"{extra}_{index}_{base_name}.html"
    return new_filename



#####################################
# FOLDER = "simple_franky"
# FOLDER="dan_carber"

# FOLDER="dan_carber3"
FOLDER="dan_carber3_250"

#####################################

base_directory="/sdf/data/neutrino/zhulcher/BNBNUMI/"
directory=base_directory+FOLDER+"_analysis/npyfiles/"

directory2=f"{base_directory}{FOLDER}_larcv_truth"


# event_display_old_path=base_directory+FOLDER+"_files_reco"
event_display_new_path = 'event_displays/'+FOLDER  # Replace with your folder path
os.makedirs(event_display_new_path, exist_ok=True)
for p in ["Kp","lambda","truth","assoc_kp_lam","k0s"]:
    for t in ["truth","reco"]:
        os.makedirs(event_display_new_path+f"/{p}/{t}", exist_ok=True)
        os.makedirs('plots/'+FOLDER+f"/{p}/{t}", exist_ok=True)
# clear_html_files(event_display_old_path); print(f"clearing from {event_display_old_path}")




def copy_and_rename_file(event_display_old_path,source_file, destination_folder, new_name,num,mode):
    """
    Copy a file to a destination folder and rename it.
    
    Parameters:
        source_file (str): Path to the source file.
        destination_folder (str): Path to the destination folder.
        new_name (str): New name for the file (including the extension).
    
    Returns:
        str: Full path to the newly created file.
    """

    destination_file = os.path.join(destination_folder, new_name)
    if os.path.exists(destination_file):#os.path.islink(destination_file) or 
        return destination_file
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Build the full path for the new file
    
    
    # Copy the file to the new location with the new name

    base_name = os.path.splitext(os.path.basename(source_file))[0]
    actual_source_file=event_display_old_path+'/'+base_name+'/eventdisplays/'+str(num)+'.html'
    if (not os.path.exists(actual_source_file)):
        print("saving",event_display_old_path+'/'+base_name,num)
        save_plotlies(event_display_old_path+'/'+base_name,num,mode)

    # assert os.path.exists(actual_source_file)
    if os.path.exists(destination_file):# or os.path.islink(destination_file):
        return destination_file
    # print("symlink",actual_source_file, destination_file)
    os.symlink(actual_source_file, destination_file)
    return destination_file


def round_to_2(x):
    if x!=x: return np.nan
    if x==0: return 0
    return round(x, -int(np.floor(np.log10(np.abs(x)))) + 3)

import textwrap
def wrap_text(text, width=15):
    return "\n".join(textwrap.wrap(str(text), width=width))

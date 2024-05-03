from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import comparison
from math import sqrt
from learn import DEFAULT_folder_name
import os
import torch
from matplotlib import cm

from models.RigidBody import Cannonical

from plot_compare import add_plot,split_to_forward_paths,get_total_errors

def prep_data(name,mode):
    df_soft_errors = pd.read_csv(name+f"/data/errors_{mode}.csv")
    train_mov_errors = df_soft_errors["train_mov"]
    validation_mov_errors = df_soft_errors["val_mov"]
    return (train_mov_errors,validation_mov_errors)

def plot_fig(args,mode):
    name1 = args.folder_name1
    name2 = args.folder_name2
    name3 = args.folder_name3

    fig, ax = plt.subplots()
    data=prep_data(name1,mode)
    comparison.add_plot(plt, y=data[0][1:], name=f"{name1.replace('_', ' ')}: {mode} train move")
    comparison.add_plot(plt, y=data[1][1:], name=f"{name1.replace('_', ' ')}: {mode} val move")

    data=prep_data(name2,mode)
    comparison.add_plot(plt, y=data[0][1:], name=f"{name2.replace('_', ' ')}: {mode} train move")
    comparison.add_plot(plt, y=data[1][1:], name=f"{name2.replace('_', ' ')}: {mode} val move")

    data=prep_data(name3,mode)
    comparison.add_plot(plt, y=data[0][1:], name=f"{name3.replace('_', ' ')}: {mode} train move")
    comparison.add_plot(plt, y=data[1][1:], name=f"{name3.replace('_', ' ')}: {mode} val move")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    
    plt.savefig(f"{name1}_{name2}_{name3}_{mode}_move.png")
    plt.show() 
    

def plot_training_errors(args):
    """
    The function `plot_training_errors` reads error data from CSV files and plots the training and validation errors for different scenarios.
    
    :param args: The `args` parameter is an object that contains the following attributes:
    """
    print("***If Runtime tkinter errors are raised, it is because some matplotlib vs threads problems. Shouldn't be serious.***")

    

    if args.soft:
        plot_fig(args,"soft")

    if args.implicit:
        plot_fig(args,"implicit")

    if args.without:
        plot_fig(args,"without")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CAN", type=str, help="Model: CANN RB or HT.")
    parser.add_argument("--implicit", default = False, action="store_true", help="Implicit")
    parser.add_argument("--without", default = False, action="store_true", help="Without")
    parser.add_argument("--soft", default = False, action="store_true", help="Soft")
    parser.add_argument("--GT", default=False, action="store_true", help="grand truth")
    parser.add_argument("--folder_name1", default=DEFAULT_folder_name, type=str, help="Folder name")
    parser.add_argument("--folder_name2", default=DEFAULT_folder_name, type=str, help="Folder name")
    parser.add_argument("--folder_name3", default=DEFAULT_folder_name, type=str, help="Folder name")
    parser.add_argument('--H',  type=str, help='Hamiltonian choice for Cannonical model. 1DHO  - 1 dimensoinal harmonic oscilator', required=False, default="1DHO")
    parser.add_argument('--plot_total_sim_error', help='plots the log norm of the difference between GT and simulated trajectories',action="store_true" , default=False)
    parser.add_argument("--export", default=True, action="store_true", help="Save figures and logs.")
    parser.add_argument("--plot_training_errors", default=False, action="store_true", help="Plot training erros.")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    dim = None
    match args.model:
        case "CANN": 
            dim = Cannonical.get_ham(args.H)[1]
        case "RB" | "HT": 
            dim = 3

    dfli1, dflw1, dfls1, dfgt1 = None, None, None, None

    if args.GT:
        file_name_GT1 = args.folder_name1+"/data/generalization.xyz"
        dfgt1 = pd.read_csv(file_name_GT1 )

    if args.implicit:
        file_name_learned_implicit1 = args.folder_name1+"/data/learned_implicit.xyz"
        dfli1 = pd.read_csv(file_name_learned_implicit1 )

    if args.soft:
        file_name_learned_soft1 = args.folder_name1+"/data/learned_soft.xyz"
        dfls1 = pd.read_csv(file_name_learned_soft1 )

    if args.without:
        file_name_learned_without1 = args.folder_name1+"/data/learned_without.xyz"
        dflw1 = pd.read_csv(file_name_learned_without1 )

    dfli2, dflw2, dfls2, dfgt2 = None, None, None, None

    if args.GT:
        file_name_GT2 = args.folder_name2+"/data/generalization.xyz"
        dfgt2 = pd.read_csv(file_name_GT2 )

    if args.implicit:
        file_name_learned_implicit2 = args.folder_name2+"/data/learned_implicit.xyz"
        dfli2 = pd.read_csv(file_name_learned_implicit2 )

    if args.soft:
        file_name_learned_soft2 = args.folder_name2+"/data/learned_soft.xyz"
        dfls2 = pd.read_csv(file_name_learned_soft2 )

    if args.without:
        file_name_learned_without2 = args.folder_name2+"/data/learned_without.xyz"
        dflw2 = pd.read_csv(file_name_learned_without2 )

    dfli3, dflw3, dfls3, dfgt3 = None, None, None, None

    if args.GT:
        file_name_GT3 = args.folder_name3+"/data/generalization.xyz"
        dfgt3 = pd.read_csv(file_name_GT3 )

    if args.implicit:
        file_name_learned_implicit3 = args.folder_name3+"/data/learned_implicit.xyz"
        dfli3 = pd.read_csv(file_name_learned_implicit3 )

    if args.soft:
        file_name_learned_soft3 = args.folder_name3+"/data/learned_soft.xyz"
        dfls3 = pd.read_csv(file_name_learned_soft3 )

    if args.without:
        file_name_learned_without3 = args.folder_name3+"/data/learned_without.xyz"
        dflw3 = pd.read_csv(file_name_learned_without3 )



    if args.plot_training_errors:
        plot_training_errors(args)


    if args.plot_total_sim_error:
        x_gt = split_to_forward_paths(dfgt1, "q1")[1]
        times = dfgt1["time"][:len(x_gt)]
        
        if args.soft:
            add_plot(plt, times, get_total_errors(dfgt1,dfls1,times,dim), name=args.folder_name1 +" soft error",log =True)
            add_plot(plt, times, get_total_errors(dfgt2,dfls2,times,dim), name=args.folder_name2 +" soft error",log =True)
            add_plot(plt, times, get_total_errors(dfgt3,dfls3,times,dim), name=args.folder_name3 +" soft error",log =True)
        if args.implicit:
            add_plot(plt, times, get_total_errors(dfgt1,dfli1,times,dim), name=args.folder_name1 +" implicit error",log =True)
            add_plot(plt, times, get_total_errors(dfgt2,dfli2,times,dim), name=args.folder_name2 +" implicit error",log =True)
            add_plot(plt, times, get_total_errors(dfgt3,dfli3,times,dim), name=args.folder_name3 +" implicit error",log =True)
        if args.without:
            add_plot(plt, times, get_total_errors(dfgt1,dflw1,times,dim), name=args.folder_name1+" without error",log =True)
            add_plot(plt, times, get_total_errors(dfgt2,dflw2,times,dim), name=args.folder_name2+" without error",log =True)
            add_plot(plt, times, get_total_errors(dfgt3,dflw3,times,dim), name=args.folder_name3+" without error",log =True)
        plt.legend()
        plt.title("Log plot of MSE of simulated trajectories")
        plt.xlabel("Time")
        if args.export:
            file_name = "./"+args.folder_name1+"_"+args.folder_name2+"_"+args.folder_name3+"_logMSE.png"
            print("Exporting figure to: "+file_name)
            plt.savefig(file_name) 
        plt.show()
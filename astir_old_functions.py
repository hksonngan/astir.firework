#!/usr/bin/env python
#
# This file is part of FIREwork
# 
# FIREwork is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIREwork is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FIREwire.  If not, see <http://www.gnu.org/licenses/>.
#
# FIREwire Copyright (C) 2008 - 2010 Julien Bert 

## Convention
# import
# constants
# shell functions
# cmd functions
# shell io
# parser and caller

#=== import =====================

from   pymir_kernel import image_read, image_write
from   pymir_kernel import image_im2mat, image_mat2im
from   pymir_kernel import image_show, image_show_get_pts
from   pymir_kernel import image_plot_points, image_plot_lines, image_show_stereo_get_pts
from   pymir_kernel import color_color2gray, color_gray2color, color_colormap
from   pymir_kernel import space_reg_ave, space_merge, space_align, space_G_transform
from   pymir_kernel import resto_wiener, image_anaglyph
from   pymir_kernel import geo_homography
from   math         import log, sqrt
from   optparse     import OptionParser
from   copy         import deepcopy
import os, sys
import readline # allow to back line in the shell
import cPickle, atexit

#=== constants ==================

listfun = ['exit', 'ls', 'rm', 'mv', 'cp', 'mem', 'save_var', 
           'load_var', 'add', 'fun', 'save_world', 'load_world',
           'ldir', 'load_im', 'save_im', 'show_mat', 'color2gray',
           'seq2mat', 'seq_reg_ave', 'load_vid', 'wiener', 'mosaicing',
           'cut_seq', 'licence', 'gray2color', 'anaglyph', 'colormap', 
           'sub', 'div', 'mul', 'info']

B  = '\033[0;34m' # blue
BC = '\033[0;36m' # blue clear (or blue sky)
G  = '\033[0;32m' # green
GB = '\033[1;32m' # green bold
R  = '\033[0;31m' # red
RB = '\033[1;31m' # red bold
N  = '\033[m'     # neutral
Y  = '\033[0;33m' # yellow

sizebar = 32
version = 'v0.36'

# WORLD structure: WORLD['keyname'] = [header, data]
# header = 'seq' or 'mat'
# data   = array(high, width, nb_channel)
WORLD  = {}

# read history
readline.set_history_length(500)
histfile = os.path.join(os.environ['HOME'], '.astir_history')
try:
    readline.read_history_file(histfile)
except IOError:
    pass
# save always before exit, even when sys.exit is raised
atexit.register(readline.write_history_file, histfile)

# errors flag: succes 1, nothing 0, error -1

#=== shell functions ============
def inbox_overwrite(name):
    answer = ''
    while answer != 'y' and answer != 'n':
        answer = raw_input('%s??%s Overwrite %s (%s[y]%s/%sn%s): '
                          % (Y, N, name, GB, N, R, N))
        if answer == '': answer = 'y'
    
    return answer

def inbox_question(message):
    answer = ''
    while answer != 'y' and answer != 'n':
        answer = raw_input('%s??%s %s (%s[y]%s/%sn%s): ' 
                          % (Y, N, message, GB, N, R, N))
        if answer == '': answer = 'y'
    
    return answer

def inbox_input(message):
    while 1:
        try:
            answer = raw_input('%s??%s %s ' % (Y, N, message))
            if answer == '':
                print '%s!!%s Again' % (B, N)
                continue
            break
        except:
            print '%s!!%s Again' % (B, N)
            continue

    return answer

def outbox_exist(name):
    print '%s!!%s %s doesn\'t exist' % (B, N, name)

def outbox_error(message):
    print '%sEE%s %s' % (R, N, message)

def outbox_bang(message):
    print '%s!!%s %s' % (B, N, message)

def check_name(names):
    if not isinstance(names, list): names = [names]
    lname = WORLD.keys()
    for name in names:
        if name not in lname:
            outbox_exist(name)
            return -1
    return 1

def check_name_file(names):
    if not isinstance(names, list): names = [names]
    lname = os.listdir('.')
    for name in names:
        if name not in lname:
            outbox_exist(name)
            return -1
    return 1
 
def check_overwrite(names):
    if not isinstance(names, list): names = [names]
    lname = WORLD.keys()
    for name in names:
        while name in lname:
            answer = inbox_overwrite(name)
            if answer == 'n': return 0
            else: break
    
    '''
    while trg in lname:
        answer = inbox_overwrite(trg)
        if answer == 'n': trg == inbox_input('Change to a new name:')
        else: break
    '''

    return 1

def check_overwrite_file(names):
    if not isinstance(names, list): names = [names]
    lname = os.listdir('.')
    for name in names:
        while name in lname:
            answer = inbox_overwrite(name)
            if answer == 'n': return 0
            else: break

def check_seq(names):
    if not isinstance(names, list): names = [names]
    lname = WORLD.keys()
    for name in names:
        if WORLD[name][0] != 'seq':
            outbox_error('Only seq varaible can be used')
            return -1
    return 1

def check_mat(names):
    if not isinstance(names, list): names = [names]
    lname = WORLD.keys()
    for name in names:
        if WORLD[name][0] != 'mat':
            outbox_error('Only mat varaible can be used')
            return -1
    return 1

def check_RGB(im):
    n = im.shape
    if len(n) == 3:
        if n[2] >= 3:
            return 1
    outbox_error('Must be in RGB format')
    return -1

def check_L(im):
    n = im.shape
    if len(n) == 2: return 1
    outbox_error('Must be in L format')
    return -1

class progress_bar:
    def __init__(self, valmax, maxbar, title):
        if valmax == 0:  valmax = 1
        if maxbar > 200: maxbar = 200
        valmax -= 1
        self.valmax = valmax
        self.maxbar = maxbar
        self.title  = title

    def update(self, val):
        sys.stdout.flush()
        if val > self.valmax: val = self.valmax
        perc  = round((float(val) / float(self.valmax)) * 100)
        scale = 100.0 / float(self.maxbar)
        bar   = int(perc / scale)
        out   = '\r%s%s %s[%s%s%s%s] %s%3d %%%s' % (BC, self.title.ljust(10), G, Y, '=' * bar, ' ' * (self.maxbar - bar), G, RB, perc, N)
        sys.stdout.write(out)
        if perc == 100: sys.stdout.write('\n')

#=== cmd functions =============
def call_ls(args):
    '''
Listing all variables in work space.
Liste toutes les variables dans l espace de travail
    '''
    usage = 'ls'
    prog  = 'ls'
    desc  = call_ls.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) > 0:
        p.print_help()
        return 0
    lname = WORLD.keys()
    lname.sort()
    space = 10 # cst columns size
    print '%s %s %s' % ('name'.ljust(space), 'type'.ljust(space), 'size'.ljust(space))
    for name in lname:
        kind = WORLD[name][0]
        if kind == 'mat':
            dim   = WORLD[name][1].shape
            dtype = WORLD[name][1].dtype
            h, w  = dim[:2]
            if len(dim) == 3:
                if   dim[2] == 3: mode = 'RGB'
                elif dim[2] == 4: mode = 'RGBA'
            else:                 mode = 'L'
            print '%s %s%s %s%s%s' % (name.ljust(space), 
              G, 'mat'.ljust(space), 
              R, '[%ix%i %s %s]' % (w, h, mode, dtype), N)
        elif kind == 'seq':
            dim   = WORLD[name][1].shape
            dtype = WORLD[name][1].dtype
            nbm   = dim[0]
            h, w  = dim[1:3]
            if len(dim) == 4:
                if   dim[3] == 3: mode = 'RGB'
                elif dim[3] == 4: mode = 'RGBA'
            else:                 mode = 'L'
            print '%s %s%s %s%s%s' % (name.ljust(space), 
              G, 'seq'.ljust(space), 
              R, '[%i mat %ix%i %s %s]' % (nbm, w, h, mode, dtype), N)

    return 1

def call_ldir(args):
    '''
Listing of the current directory.
Liste du dossier courant
    '''
    usage = 'ldir'
    prog  = 'ldir'
    desc  = call_ldir.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) > 0:
        p.print_help()
        return 0
    os.system('ls')
    return 1

def call_rm(args):
    '''
Remove variables in work space.
Efface des variables dans l espace de travail
    '''
    usage = 'rm <name>\nrm <name1> <name2>\nrm <na*>\nrm <*>'
    prog  = 'rm'
    desc  = call_rm.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) == 0:
        p.print_help()
        return 0
    if   args[0] == '*': args = WORLD.keys()
    elif args[0].find('*') != -1:
        lname   = WORLD.keys()
        pattern = args[0].split('*')
        if len(pattern) != 2:
            outbox_error('Bad pattern with the joker *')
            return -1
        args = []
        for name in lname:
            if name.find(pattern[0]) != -1 and name.find(pattern[1]) != -1:
                args.append(name)
        if len(args) == 0:
            outbox_error('No variable matchs with the pattern *')
            return -1
        args.sort()
        outbox_bang('%i variables match with the pattern' % len(args))
        print args
        answer = inbox_question('Agree to remove all of them')
        if answer == 'n': return 0

    if not check_name(args): return -1
    for name in args: del WORLD[name]

    return 1

def call_mv(args):
    '''
Move/rename variable.
Deplace/renomme une variable
    '''
    usage = 'mv <source_name> <target_name>\nmv im0 im1'
    prog  = 'mv'
    desc  = call_ls.__doc__
    p = OptionParser(description = desc, prog=prog, version=version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 2:
        p.print_help()
        return 0
    src, trg = args
    if not check_name(src):      return -1
    if not check_overwrite(trg): return  0
    WORLD[trg] = deepcopy(WORLD[src])
    del WORLD[src]
    del data

    return 1

def call_cp(args):
    '''
Copy variable
Copie une variable
    '''
    usage = 'cp <source_name> <target_name>\ncp im0 im1'
    prog  = 'cp'
    desc  = call_cp.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 2:
        p.print_help()
        return 0
    src, trg = args
    if not check_name(src):      return -1
    if not check_overwrite(trg): return  0
    WORLD[trg] = deepcopy(WORLD[src])
    del data

    return 1

def call_mem(args):
    '''
Memories used in work space by the variables
Mémoire utilisee dans les espaces de travails par les variables
    '''
    usage = 'mem'
    prog  = 'mem'
    desc  = call_mem.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) > 0:
        p.print_help()
        return 0
    space = 10
    txt = ['', 'k', 'M', 'G', 'T']
    nbb = {'float8':1, 'float16':2, 'float32':4, 'float64':8,
            'uint8':1,  'uint16':2,  'uint32':4,  'uint64':8}
    lname = WORLD.keys()
    for name in lname:
        size   = WORLD[name][1].size
        dtype  = WORLD[name][1].dtype
        size  *= nbb[dtype]
        ie     = int(log(size) // log(1e3))
        size  /= (1e3 ** ie)
        size   = '%5.2f %sB' % (size, txt[ie])
        print '%s %s%s %s%s%s' % (name.ljust(space), 
              G, kind.ljust(space), 
              R, size.ljust(space), N)
        
    return 1

def call_fun(args):
    '''
Listing funtions available in Astir
Liste les fonctions disponible dans Astir
    '''
    usage = 'fun'
    prog  = 'fun'
    desc  = call_fun.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    p.add_option('-c', action='store', type='int', default='4', help='Number of columns. Nombre de colonnes')
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) > 0:
        p.print_help()
        return 0
    listfun.sort()
    sfun = len(listfun)
    nc   = opt.c
    if sfun % nc == 0: nl = sfun // nc
    else:              nl = (sfun // nc) + 1
    smax = 0
    for i in xrange(sfun):
        val = len(listfun[i])
        if val > smax: smax = val
    for i in xrange(nl):
        txt = ''
        for j in xrange(nc):
            ind = j * nl + i
            if ind < sfun: txt += '%s  ' % listfun[ind].ljust(smax)
            else:          txt += ''
        print txt

    return 1

def call_save_var(args):
    '''
Save Astir variable to file.
Sauvegarde une variable Astir dans un fichier
    '''
    usage = 'save_var <var_name> <file_name>\nsave_var im1 image1.pck'
    prog  = 'save_var'
    desc  = call_save_var.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 2:
        p.print_help()
        return 0
    name, fname = args
    if not check_name_file(name):       return -1
    if not check_overwrite_file(fname): return -1
    f = open(fname, 'w')
    local = ['var_astir', name, WORLD[name]]
    cPickle.dump(local, f, 1)
    f.close()

    return 1

def call_save_world(args):
    '''
Save the whole work space to a file.
Sauvegarde entierement l espace de travail dans un fichier
    '''
    usage = 'save_world <file_name>\nsave_world backup.pck'
    prog  = 'save_world'
    desc  = call_save_world.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 1:
        p.print_help()
        return 0
    kname = WORLD.keys()
    if len(kname) == 0:
        outbox_bang('Nothing to save')
        return 0
    fname = args[0]
    if not check_overwrite_file(fname): return -1
    f = open(fname, 'w')
    local = ['world_astir', WORLD]
    cPickle.dump(local, f, 1)
    f.close()
    
    return 1

def call_load_var(args):
    '''
Load a variable from a file to work space.
Charge une variable depuis un fichier dans l espace de travail.
    '''
    usage = 'load_var <file_name>\nload_var mydata.pck'
    prog  = 'load_var'
    desc  = call_load_var.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 1:
        p.print_help()
        return 0
    fname = args[0]
    if not check_name_file(fname): return -1
    f = open(fname, 'r')
    try: local = cPickle.load(f)
    except:
        outbox_error('Can not open the file')
        f.close()
        return -1
    f.close()
    if local[0] != 'var_astir':
        outbox_error('Not Astir format')
        return -1
    varname = local[1]
    vardata = local[2]
    lname   = WORLD.keys()
    while varname in lname:
        answer = inbox_overwrite(varname)
        if answer == 'n': varname = inbox_input('Change to a new name:')
        else: break
    WORLD[varname] = vardata

    return 1

def call_load_world(args):
    '''
Load a work space from a file.
Charge un espace de travial depuis un fichier.
    '''
    usage = 'load_world <file_name>\nload_world mydata.pck'
    prog  = 'load_world'
    desc  = call_load_world.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 1:
        p.print_help()
        return 0
    fname = args[0]
    if not check_name_file(fname): return -1
    f = open(fname, 'r')
    try: local = cPickle.load(f)
    except:
        outbox_error('Can not open the file')
        f.close()
        return -1
    f.close()
    if local[0] != 'world_astir':
        outbox_error('Not Astir format')
        return -1
    answer = inbox_question('All variables will be deleted, are you agree')
    if answer == 'n': return 0
    del WORLD
    WORLD = local[1]

    return 1

def call_load_im(args):
    '''
Load images from files.
Chared des images depuis des fichiers.
    '''
    usage = 'load_im <file_name.[bmp, jpg, png, tiff]>\nload_im file_na*.png'
    prog  = 'load_im'
    desc  = call_load_im.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) == 0:
        p.print_help()
        return 0

    if args[0].find('*') != -1:
        lname   = os.listdir('.')
        pattern = args[0].split('*')
        if len(pattern) != 2:
            outbox_error('Bad pattern with joker *')
            return -1
        mem = []
        for name in lname:
            if name.find(pattern[0]) != -1 and name.find(pattern[1]) != -1:
                mem.append(name)
        if len(mem) == 0:
            outbox_error('No image matchs with the pattern')
            return -1
        fname = mem[0]
        mem.sort()
        outbox_bang('%i files match with the pattern' % len(mem))
        print mem
        answer = inbox_question('Agree to load all of them')
        if answer == 'n': return 0
    else:
        mem   = None
        fname = args[0]

    buf = fname.split('.')
    if len(buf) == 2: name, ext = fname.split('.')
    else:             name, ext = None, None
    if ext not in ['bmp', 'jpg', 'png', 'tiff']:
        outbox_error('Bad extension (bmp, jpg, png or tiff)')
        return -1
    if not check_name_file(fname): return -1
    if not check_overwrite(name):  return  0
    if mem is None:
        im  = image_read(fname)
        WORLD[name] = ['mat', im]
        del im
    else:
        bar  = progress_bar(len(mem), sizebar, 'loading')
        seq  = []
        name = mem[0].split('.')[0]
        i    = 0
        for item in mem:
            im  = image_read(item)
            seq.append(im)
            bar.update(i)
            i += 1
        del im
        seq = array(seq)
        WORLD[name] = ['seq', seq]
        del seq

    return 1

def call_save_im(args):
    '''
Save image(s) from a variable (mat/seq) to a file(s).
Sauvegarde une ou des images depuis une variable (mat/seq) vers un ou des fichiers
save_im <mat_name> <file_name.[bmp, jpg, png]>
    '''
    usage = 'save_im <var_name> <file_name.[bmp, jpg, png, tiff]>\nsave_im im0 im.png\nsave_im vid im.png'
    prog  = 'save_im'
    desc  = call_save_im.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 2:
        p.print_help()
        return 0
    name  = args[0]
    fname = args[1]
    if not check_name(name): return -1
    lext = ['jpg', 'png', 'bmp', 'tiff']
    if len(fname.split('.')) != 2:
        outbox_error('File name must have an extension')
        return -1
    [fname, ext] = fname.split('.')
    if ext not in lext:
        outbox_error('Wrong extension, only jpg, png, bmp or tiff')
        return -1
    kind = WORLD[name][0]
    if kind == 'mat':
        fname = fname + '.' + ext
        if not check_overwrite_file(fname): return -1
        im = WORLD[name][1]
        image_write(im, fname)
        del im, fname
    elif kind == 'seq':
        nb    = WORLD[name][1].shape[0]
        names = [fname + '_%04i.' % i + ext for i in xrange(nb)]
        if not check_overwrite_file(names): return -1
        bar = progress_bar(nb, sizebar, 'writing')
        for i in xrange(nb):
            im = WORLD[name][1][i]
            image_write(im, names[i])
            bar.update(i)
        del im, bar, nb, names

    return 1

def call_show_mat(args):
    '''
Display a mat variable as an image.
Affiche une variable de type mat comme une image.
    '''
    usage = 'show_mat <mat_name>\nshow_mat <mat_name1> <mat_name2>'
    prog  = 'show_mat'
    desc  = call_show_mat.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) == 0 or len(args) > 2:
        p.print_help()
        return 0
 
    list_im = []
    if not check_name(args): return -1
    if not check_mat(args):  return -1    
    for name in args:
        im = WORLD[name][1]
        list_im.append(im)

    image_show(list_im)
    del list_im, args, im

    return 1

def call_color2gray(args):
    '''
Convert mat/seq color (RGB or RGBA) to gray scale (Luminance).
Convertie mat/seq en couleur (RGB ou RGBA) en niveau de gris (Luminance).
    '''
    usage = 'Convert in place\ncolor2gray <mat_name>\nConvert to new mat\n\
             color2gray <mat_name> <mat_new_name>\nConvert a mat sequence in-place\n\
             color2gray <seq_name>\nConvert a mat sequence to a new one\n\
             color2gray <seq_name> <seq_new_name>'
    prog  = 'color2gray'
    desc  = call_save_im.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) == 0 or len(args) > 2:
        p.print_help()
        return 0
 
    if len(args) == 2:
        src, trg = args
    else:
        src, trg = args[0], args[0]
    if not check_name(src): return -1
    kind  = WORLD[src][0]
    if kind == 'mat':
        im = WORLD[src][1]
        if not check_L(im): return -1
        nim = color_color2gray(im)
        WORLD[trg] = ['mat', nim]
        del nim, im
    else:
        im0  = WORLD[src][1][0]
        if not check_L(im0): return -1
        nb   = WORLD[src][1].shape[0]
        bar  = progress_bar(nb, sizebar, 'Processing')
        data = []
        for n in xrange(nb):
            nim = color_color2gray(WORLD[src][1][n])
            data.append(nim)
            bar.update(n)
        data = array(data)
        WORLD[trg] = ['seq', data]
        del data, nim, im0, bar

    return 1

def call_gray2color(args):
    '''
Convert mat/seq gray scale (Luminance) to color (RGB).
Converti mat/seq en niveau de gris (Luminance) en couleur (RGB).
    '''
    usage = 'Convert in-place\n\
             gray2color <mat_name>\n\
             Convert to new mat\n\
             gray2color <mat_name> <mat_new_name>\n\
             Convert a mat sequence in-place\n\
             gray2color <seq_name>\n\
             Convert a mat sequence to a new one\n\
             gray2color <seq_name> <seq_new_name>\n'
    prog  = 'gray2color'
    desc  = call_gray2color.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) == 0 or len(args) > 2:
        p.print_help()
        return 0
    if len(args) == 2: src, trg = args
    else:              src, trg = args[0], args[0]
    if not check_name(src): return -1
    if kind == 'mat':
        im  = WORLD[src][1]
        if not check_L(im): return -1
        nim = color_gray2color(im)
        WORLD[trg] = ['mat', nim]
        del nim, im
    else:
        im0 = WORLD[src][1][0]
        if not check_L(im0): return -1
        nb  = WORLD[src][1].shape[0]
        bar  = progress_bar(nb, sizebar, 'Processing')
        data = []
        for n in xrange(nb):
            nim = color_gray2color(WORLD[src][1][n])
            data.append(nim)
            bar.update(n)
        data = array(data)
        WORLD[trg] = ['seq', data]
        del data, nim, im0, bar

    return 1

def call_seq2mat(args):
    '''
Extract mat variables from a sequence.
Extrait mar variables depuis une sequence.
    '''
    usage = 'seq2mat <seq_name> <mat_name> [options]\n\
             Extract number 5\n\
             seq2mat vid im -i 5\n\
             Extract mat 5 to 10\n\
             seq2mat vid im -s 5 -e 10\n\
             Extract mat 5 through to the end\n\
             seq2mat vid im -s 5\n\
             Extract first mat through to the 10th (include)\n\
             seq2mat vid im -e 10\n'
    prog  = 'seq2mat' 
    desc  = call_seq2mat.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.add_option('-i', action='store', type='int', default='-1', help='Extract mat i')
    p.add_option('-s', action='store', type='int', default='-1', help='Extract starting number')
    p.add_option('-e', action='store', type='int', default='-1', help='Extract stoping number')
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 2:
        p.print_help()
        return 0
    if opt.i != -1 and (opt.s  != -1 or opt.e != -1):
        outbox_error('Choose option i OR e and s, not both')
        return -1
    if opt.i == -1 and opt.s == -1 and opt.e == -1: opt.i = 0
    src, trg = args
    if not check_name(src): return -1
    if not check_seq(src):  return -1
    if opt.i != -1: opt.s = opt.e = opti
    else:
        if opt.s == -1: opt.s = 0
        if opt.e == -1: opt.e = WORLD[src][1].shape[0]
    names = [trg + '_%04i' % i for i in xrange(opt.s, opt.e + 1)]
    if not check_overwrite(names): return -1
    n = 0
    for i in xrange(opt.s, opt.e + 1):
        im = WORLD[src][1][i]
        WORLD[name[n]] = ['mat', im]
        n += 1

    return 1

def call_seq_reg_ave(args):
    '''
This function use a simple registration to match images together
and compute the averages. Which will increase the signal-noise ratio.
Cette fonction permet de recaller les images entre elles
afin de calculer la moyenne. Qui vat augmenter le rapport signal sur bruit.
    '''
    usage = 'seq_reg_ave <seq_name> [option]\n\
             seq_reg_ave im\n\
             seq_reg_ave im -d 10 -w 35 -o res'
    prog  = 'seq_reg_ave'
    desc  = call_seq_reg_ave.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    p.add_option('-d', action='store', type='int',    default='10',      help='dx/dy is the translation range search on x/y (x-dx to x+dx) (default 10)')
    p.add_option('-w', action='store', type='int',    default='35',      help='window size used to track translation between images (must be odd) (default 35)')
    p.add_option('-o', action='store', type='string', default='res_ave', help='output name (default res_ave)')
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 1:
        p.print_help()
        return 0
    src   = args[0]
    if not check_name(src):        return -1
    if not check_seq(src):         return -1
    if not check_overwrite(opt.o): return 0
    dx = dy = p.d
    ws = p.w
    if ws % 2 == 0:
        ws += 1
        outbox_bang('Window size must be odd, set to %i' % ws)
    dw = (ws - 1) // 2
    im = WORLD[src][1][0]
    # TODO change this part with new kernel
    p   = image_show_get_pts(im, 1, rad = dw)
    print 'point selected:', p[0]
    ave = space_reg_ave(WORLD[src][1], p[0], ws, dx, dy)
    WORLD[opt.o] = ['mat', ave]

    return 1

def call_load_vid(args):
    '''
Load video (avi file only) as a sequence
Charge une video (fichier avi) en tant qu une sequence
load_vid <video_name> <frame_per_second>
    '''
    usage = 'load_vid <video_name> [option]\n\loav_vid neptune.avi -f 10'
    prog  = 'load_vid'
    desc  = call_load_vid.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    p.add_option('-f', action='store', type='int', default='10', help='frame per second (default 10)')
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 1:
        p.print_help()
        return 0

    src  = args[0]
    freq = opt.f
    if not check_name_file(src): return -1
    name, ext = filename.split('.')
    if ext != 'avi':
        outbox_error('Must be an avi file')
        return -1
    if not check_overwrite(name): return 0
    print 'Extract images...'
    pattern = '.tmp_astir_'
    try:
        os.system('ffmpeg -i %s -r %i -f image2 "%s%%4d.png"' % (filename, freq, pattern))
    except:
        outbox_error('Impossible to extract images from the video')
        return -1
    lname = os.listdir('.')
    mem   = []
    for file in lname:
        if file.find(pattern) != -1: mem.append(file)
    bar = progress_bar(len(mem), sizebar, 'loading')
    seq = []
    i   = 0
    mem.sort()
    for item in mem:
        im  = image_read(item)
        seq.append(im)
        bar.update(i)
        i += 1
    seq = array(seq)
    WORLD[name] = ['seq', seq]
    os.system('rm -f %s*' % pattern)
    del im, seq, bar
    
    return 1

def call_wiener(args):
    '''
Image restoration by Wiener filter.
Restauration d image par filtre de Wiener
wiener <mat_source_name> <mat_res_name>
    '''
    usage = 'wiener <mat_source_name> <mat_res_name>'
    prog  = 'wiener'
    desc  = call_wiener.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 1:
        p.print_help()
        return 0
    src   = args[0]
    trg   = args[1]
    if not check_name(src):      return -1
    if not check_mat(src):       return -1
    if not check_overwrite(trg): return -1
    res = resto_wiener(WORLD[src][1])
    WORLD[trg] = ['mat', res]

    return 1

def call_mosaicing(args):
    '''
Create mosaicing from two images
mosaicing <mat_1> <mat_2>
    '''
    ## TODO parser mosaicing
    mat1   = WORLD[args[0]][1]
    mat2   = WORLD[args[1]][1]
    ch     = len(mat1)
    ws     = 35
    im1    = image_mat2im(mat1)
    im2    = image_mat2im(mat2)
    if   ch == 1:
        im1c = color_gray2color(im1)
        im2c = color_gray2color(im2)
        im1g = im1
        im2g = im2
    elif ch == 3:
        im1g = color_color2gray(im1)
        im2g = color_color2gray(im2)
        im1c = im1
        im2c = im2

    p1, p2 = image_show_stereo_get_pts(im1c, im2c, 4)
    print p2
    for n in xrange(len(p1)):
        print 'Aligned match points %i' % n
        xp, yp = space_align(im1g, im2g, p1[n], 35, 5, 5, p2[n])
        p2[0][0] = p1[0][0] + yp
        p2[0][1] = p1[0][1] + xp
    print p2
    
    sys.exit()
    H = geo_homography(p1, p2)

    res, l, t = space_G_transform(H, im2, 'NEAREST')
    print p1
    print p2
    print H.I
    #res = space_merge(mat1, mat2, p1, p2, 'ada')
    WORLD['res'] = ['mat', res]

    return 1

def call_cut_seq(args):
    '''
Cut a part of sequence to a new one, start and stop
specifies the part you want keep. Coupe une partie d une 
sequence dans une nouvelle, start et stop specifis la partie
que vous voulez garder.
    '''
    usage = 'cut_seq <seq_name> <new_seq_name> [option]\n\
             cut_seq vid1 newvid -s 10 -e 34\n'
    prog  = 'cut_seq'
    desc  = call_cut_seq.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    p.add_option('-s', action='store', type='int', default='-1', help='Start number (default 0)')
    p.add_option('-e', action='store', type='int', default='-1', help='Stop number (default until the end)')
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 2:
        p.print_help()
        return 0
    src, trg = args
    if not check_name(src):      return -1
    if not check_seq(src):       return -1
    if not check_overwrite(trg): return  0
    if opt.s == -1: opt.s = 0
    if opt.e == -1: opt.e = WORLD[src][1].shape[0]
    seq = []
    for n in xrange(opt.s, opt.e + 1):
        seq.append(WORLD[src][1][n])
    seq = array(seq)
    WORLD[trg] = ['seq', seq]
    del seq

    return 1

def call_licence(args):
    data = open('COPYING', 'r').readlines()
    for line in data: print line.strip('\n')

    return 1

def call_anaglyph(args):
    '''
Create an anaglyph image from two RGB matrix (right and left).
Creer une image anaglyphe depuis deux mat RGB (droite et gauche).
    '''
    usage = 'anaglyph <mat_right> <mat_left> [options]\n\
             anaglyph imr img -o newim\n'
    prog  = 'anaglyph'
    desc  = call_anaglyph.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    p.add_option('-o', action='store', type='string', default='res_anag', help='Output name (default res_anag)')
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 2:
        p.print_help()
        return 0
    src1, src2 = args
    trg        = opt.o
    if not check_name([src1, src2]): return -1
    if not check_mat([src1, src2]):  return -1
    if not check_overwrite(trg):     return 0
    im1, im2 = WORLD[src1][1], WORLD[src2][1]
    if not check_RGB(im1): return -1
    if not check_RGB(im2): return -1
    res = image_anaglyph(im1, im2)
    WORLD[trg] = ['mat', res]
    
    return 1

def call_colormap(args):
    '''
Apply false-colors to a luminance mat.
Applique des fausses couleurs sur une mat en luminance
colormap <mat_name> <kind_of_map> <new_mat_name>
different color of map: jet, hsv, hot

colormap im1 hot im_map    
    '''
    usage = 'colormap <mat_name> <new_name> [options]\n\
             colormap im1 im1color -c jet'
    prog  = 'colormap'
    desc  = call_colormap.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    p.add_option('-c', action='store', type='string', default='jet', help='Kind of colormap jet, hsv and hot (default is jet)')
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 2:
        p.print_help()
        return 0
    src, trg = args
    kind     = opt.c
    if not check_name(src):      return -1
    if not check_mat(src):       return -1
    if not check_overwrite(trg): return  0
    if not check_L(src):         return -1
    if kind not in ['jet', 'hsv', 'hot']:
        outbox_error('Kind of map color unknown')
        return -1
    res = color_colormap(WORLD[src][1], kind)
    WORLD[trg] = ['mat', res]

    return 1

def call_add(args):
    '''
Add two mat variables (L or RGB).
Ajoute deux varaible mat (L ou RGB)
mat_c = mat_a + mat_b
    '''
    usage = 'add <mat_a> <mat_b> <mat_c>\n\
             add im1 im2 res\n'
    prog  = 'add'
    desc  = call_add.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 3:
        p.print_help()
        return 0
    src1, src2, trg = args
    if not check_name([src1, src2]): return -1
    if not check_mat([src1, src2]):  return -1
    if not check_overwrite(trg):     return 0
    mat1 = WORLD[src1][1]
    mat2 = WORLD[src2][1]
    res  = mat1 + mat2
    WORLD[trg] = ['mat', res]

    return 1

def call_sub(args):
    '''
Substract two mat variables (L or RGB).
Soustract deux variables mat (L ou RGB)
mat_c = mat_a - mat_b
    '''
    usage = 'sub <mat_a> <mat_b> <mat_c>\n\
             sub im1 im2 res\n'
    prog  = 'sub'
    desc  = call_sub.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 3:
        p.print_help()
        return 0
    src1, src2, trg = args
    if not check_name([src1, src2]): return -1
    if not check_mat([src1, src2]):  return -1
    if not check_overwrite(trg):     return 0
    mat1 = WORLD[src1][1]
    mat2 = WORLD[src2][1]
    res  = mat1 - mat2
    WORLD[trg] = ['mat', res]

    return 1

def call_mul(args):
    '''
Multiply two mat variables (L or RGB). 
Multiplie deux variables mat (L ou RGB)
mat_c = mat_a * mat_b
    '''
    usage = 'mul <mat_a> <mat_b> <mat_c>\n\
             mul im1 im2 res\n'
    prog  = 'mul'
    desc  = call_mul.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 3:
        p.print_help()
        return 0
    src1, src2, trg = args
    if not check_name([src1, src2]): return -1
    if not check_mat([src1, src2]):  return -1
    if not check_overwrite(trg):     return 0
    mat1 = WORLD[src1][1]
    mat2 = WORLD[src2][1]
    res  = mat1 * mat2
    WORLD[trg] = ['mat', res]

    return 1

def call_div(args):
    '''
Divide two mat variables (L or RGB).
Divise deux variables mat (L ou RGB)
mat_c = mat_a / mat_b
    '''
    usage = 'div <mat_a> <mat_b> <mat_c>\n\
             div im1 im2 res\n'
    prog  = 'div'
    desc  = call_div.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 3:
        p.print_help()
        return 0
    src1, src2, trg = args
    if not check_name([src1, src2]): return -1
    if not check_mat([src1, src2]):  return -1
    if not check_overwrite(trg):     return 0
    mat1 = WORLD[src1][1]
    mat2 = WORLD[src2][1]
    res  = mat1 / mat2
    WORLD[trg] = ['mat', res]

    return 1

def call_info(args):
    '''
Return informations about a variable (size, stats, format, ...).
Retourne des informations a propos d une variable (taille, stats, format, ...)
    '''
    ## TODO info on sequence
    usage = 'info <mat_name>\n\
             info mat1\n'
    prog  = 'info'
    desc  = call_info.__doc__
    p = OptionParser(description = desc, prog = prog, version = version)
    p.set_usage(usage)
    try: opt, args = p.parse_args(args)
    except: return 0
    if len(args) != 1:
        p.print_help()
        return 0
    src   = args[0]
    if not check_name(src): return -1
    if not check_mat(src):  return -1
    mat = WORLD[src][1]
    c1, c2, c3  = G, B, Y
    print 'Name: %s%s%s Type: %s%s%s' % (c1, src, N, c1, 'mat', N)
    if   len(mat) == 1: mode = 'L'
    elif len(mat) == 3: mode = 'RGB'
    elif len(mat) == 4: mode = 'RGBA'
    print 'Mode: %s%s%s Size: %s%ix%i%s Format: %s%s%s' % (c1, mode, N, c1, mat[0].shape[1], mat[0].shape[0], N, c1, mat[0].dtype, N)
    print ''
    for c in xrange(len(mat)):
        print 'Channel %s%i%s' % (c2, c, N)
        min  = mat[c].min()
        max  = mat[c].max()
        mean = mat[c].mean()
        var  = mat[c] - mean
        var *= var
        var  = var.sum()
        var /= float(mat[c].size)
        std  = sqrt(var)
        print 'min: %s%5.3f%s max: %s%5.3f%s mean: %s%5.3f%s var: %s%5.3f%s std: %s%5.3f%s' % (c3, min, N, c3, max, N, c3, mean, N, c3, var, N, c3, std, N)

    return 1

'''
#=== documentation ==============
print '# ls'
print call_ls.__doc__
print '# ldir'
print call_ldir.__doc__
print '# rm'
print call_rm.__doc__
print '# mv'
print call_mv.__doc__
print '# cp'
print call_cp.__doc__
print '# mem'
print call_mem.__doc__
print '# fun'
print call_fun.__doc__
print '# save_var'
print call_save_var.__doc__
print '# save_world'
print call_save_world.__doc__
print '# load_var'
print call_load_var.__doc__
print '# load_world'
print call_load_world.__doc__
print '# load_im'
print call_load_im.__doc__
print '# save_im'
print call_save_im.__doc__
print '# show_mat'
print call_show_mat.__doc__
print '# color2gray'
print call_color2gray.__doc__
print '# gray2color'
print call_gray2color.__doc__
print '# colormap'
print call_colormap.__doc__
print '# seq2mat'
print call_seq2mat.__doc__
print '# seq_reg_ave'
print call_seq_reg_ave.__doc__
print '# load_vid'
print call_load_vid.__doc__
print '# wiener'
print call_wiener.__doc__
print '# mosaicing'
print call_mosaicing.__doc__
print '# cut_seq'
print call_cut_seq.__doc__
print '# add'
print call_add.__doc__
print '# sub'
print call_sub.__doc__
print '# mul'
print call_mul.__doc__
print '# div'
print call_div.__doc__
print '# info'
print call_info.__doc__
sys.exit()
'''

#=== shell io ===================

# script kernel
script_flag = False
script_end  = False
if len(sys.argv) != 1:
    script_name = sys.argv[1]
    dummy, ext  = script_name.split('.')
    if ext != 'sas':
        outbox_error('This file %s is not a Script Astir Shell (.sas).' % script_name)
        sys.exit()
    script_flag = True
    list_cmd = open(script_name, 'r').readlines()

# if mode shell
if script_flag:
    print '** Script Astir Shell V0.36 **'
else:
    print '  ___      _   _'
    print ' / _ \    | | (_)'         
    print '/ /_\ \___| |_ _ _ __' 
    print '|  _  / __| __| | \'__)'
    print '| | | \__ \ |_| | |'
    print '\_| |_/___/\__|_|_|'
    print ''
    print 'Astir  Copyright (C) 2008  Julien Bert'
    print 'This program comes with ABSOLUTELY NO WARRANTY; for details type "licence".'
    print 'This is free software, and you are welcome to redistribute it'
    print 'under certain conditions; type "licence" for details.'
    print 'GNU General Public License version 3'
    print ''
    print '** Astir Shell V0.36 **\n'


ct_cmd = 1
while 1 and not script_end:
    if script_flag:
        cmd = list_cmd[ct_cmd - 1]
        if cmd[0] == '#':
            ct_cmd += 1
            continue
        print '%s%s%s' % (B, cmd.strip('\n'), N)
        if ct_cmd == len(list_cmd):
            script_end = True
    else:
        try: cmd = raw_input('%sastir%s %i%s %%%s ' % (B, GB, ct_cmd, G, N))
        except:
            print '\nbye'
            sys.exit(0)

    if not cmd: continue

    ct_cmd   += 1
    parse     = cmd.split()
    progname  = parse[0]
    args      = parse[1:]

    if progname not in listfun:
        try: print eval(cmd)
        except:
            outbox_bang(' 8-/')
            continue

    if progname == 'exit':
        print 'bye'
        sys.exit(0)

    # caller
    eval('call_%s(args)' % progname)


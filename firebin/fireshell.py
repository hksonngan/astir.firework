#!/usr/bin/env pythonfx
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
# along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
#
# FIREwork Copyright (C) 2008 - 2011 Julien Bert 

from sys import exit

try:
    from IPython.Shell import IPShellEmbed
    from IPython       import ipapi, Prompts
except:
    print 'Please install "ipython".'
    exit()
try:
    from firework      import *
except:
    print 'Please check your installation'
    exit()
    
def myinputprompt(self, cont):
    ip     = self.api
    count  = str(len(ip.user_ns['_ih']))
    colors = Prompts.PromptColors[''].colors
    if cont:
        return '%s%s%s%%%s ' % ('\033[0;34m', '.' * (len('fw ')+ len(count) + 1), colors.in_number, colors.normal)
    else:
        return '%sfw %s%s %%%s ' % ('\033[0;34m', colors.in_number, count, colors.normal)

def myoutputprompt(self):
    ip     = self.api
    count  = str(len(ip.user_ns['_ih']))
    colors = Prompts.PromptColors[''].colors
    return '%s%s%%%s ' % (colors.out_number, ' ' * (len('fw ') + len(count) + 1), colors.normal)

'''
print 'This file is part of FIREwork'
print 'FIREwork  Copyright (C) 2008 - 2011  Julien Bert'
print 'This program comes with ABSOLUTELY NO WARRANTY; for details type "licence".'
print 'This is free software, and you are welcome to redistribute it'
print 'under certain conditions; type "licence" for details.'
print 'GNU General Public License version 3'
print ''
'''

ipshell = IPShellEmbed(banner="** FireShell V0.2 **")
ipapi.get().set_hook("generate_prompt", myinputprompt)
ipapi.get().set_hook("generate_output_prompt", myoutputprompt)
ipshell()






    


	

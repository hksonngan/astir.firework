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

include ./firework.conf

.PHONY : all $(MODULE)

all : $(MODULE)

$(MODULE) :
	@echo ':::' $@
	@make -C $@
ifeq ($(CUDA),ON)
	@make cuda -C $@
endif
	@echo ''

clean :
	@for dir in $(MODULE); do \
		make clean -C $$dir; \
	done





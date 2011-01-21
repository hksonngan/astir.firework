// This file is part of FIREwork
// 
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// FIREwork Copyright (C) 2008 - 2011 Julien Bert 

#include <stdio.h>

void convert_root2bin(char* src, char* trg) {
	Int_t crystalID1;
	Int_t crystalID2;
	Int_t rsectorID1;
	Int_t rsectorID2;
	Float_t globalPosX1;
	Float_t globalPosX2;
	Float_t globalPosY1;
	Float_t globalPosY2;      
	Float_t globalPosZ1;
	Float_t globalPosZ2;
	Double_t time1;
	Double_t time2; 
	
	Int_t nbytes = 0;
	Int_t nbevents;
	Int_t i;
	FILE * output;
	
	// prepare output
	output = fopen(trg, "ab");
	// Open tree file
	TFile *f = new TFile(src);
	TTree *T = (TTree*)f->Get("Coincidences");
	// Bind variables
	T->SetBranchAddress("crystalID1", &crystalID1);
	T->SetBranchAddress("crystalID2", &crystalID2);
	T->SetBranchAddress("rsectorID1", &rsectorID1);
	T->SetBranchAddress("rsectorID2", &rsectorID2);
	T->SetBranchAddress("globalPosX1", &globalPosX1);
	T->SetBranchAddress("globalPosX2", &globalPosX2);
	T->SetBranchAddress("globalPosY1", &globalPosY1);
	T->SetBranchAddress("globalPosY2", &globalPosY2);
	T->SetBranchAddress("globalPosZ1", &globalPosZ1);
	T->SetBranchAddress("globalPosZ2", &globalPosZ2);
	T->SetBranchAddress("time1", &time1);
	T->SetBranchAddress("time2", &time2);
	
	Int_t nbevents = T->GetEntries();
	printf("Number of events: %i\n", nbevents);
	// Read leafs
	//nbevents = 10000;
	for (i=0; i<nbevents; ++i) {
		nbytes += T->GetEntry(i);
		//fprintf(output, "%i %i %i %i %f %f %f %f %f %f %le %le\n", crystalID1, crystalID2, rsectorID1, rsectorID2, globalPosX1, globalPosX2, globalPosY1, globalPosY2, globalPosZ1, globalPosZ2, time1, time2);
		//fprintf(output, "%f %f %f %f %f %f %le %le\n", globalPosX1, globalPosY1, globalPosZ1, globalPosX2, globalPosY2, globalPosZ2, time1, time2);
		//fprintf(output, "%i %i %i %i\n", crystalID1, rsectorID1, crystalID2, rsectorID2);
		fwrite(&crystalID1, sizeof(int), 1, output);
		fwrite(&rsectorID1, sizeof(int), 1, output);
		fwrite(&crystalID2, sizeof(int), 1, output);
		fwrite(&rsectorID2, sizeof(int), 1, output);
	}
	fclose(output);
}


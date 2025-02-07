#!/usr/bin/env python3

import argparse

def acsense_to_txt(inputfname, outputfname):
	linectr = 0
	outfd = open(outputfname, 'w')
	with open(inputfname, 'rb') as infd:
		while infd:
			line = infd.read(64)
			if not line:
				break;

			cleaned = line.decode('ascii').strip('\x00\t\n ')
			if cleaned:
				outfd.write(cleaned + '\n')

def run_cfg2txt():
	parser = argparse.ArgumentParser(description='Covert AcSense Config File from acsense format to txt')
	parser.add_argument('input', help='AcSense Config File')
	parser.add_argument('output', help='Text File')
	args = parser.parse_args()

	acsense_to_txt(args.input, args.output)
	

def txt_to_acsense(inputfname, outputfname):
	linectr = 0
	outfd = open(outputfname, 'wb')
	for line in open(inputfname, 'r'):
		bstr = line.strip().encode('ascii')
		outstr = bstr + b'\x00'*(64-len(bstr))
		outfd.write(outstr)
		linectr += 1

	for i in range(linectr, 255):
		outfd.write(b'\x00'*64)

def run_txt2cfg():
	parser = argparse.ArgumentParser(description='Covert Text Config File from txt to AcSense format')
	parser.add_argument('input', help='Text File')
	parser.add_argument('output', help='AcSense Config File')
	args = parser.parse_args()

	txt_to_acsense(args.input, args.output)

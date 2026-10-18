package main

import (
	"bytes"
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
)

// exports lists the symbols a shared library defines, by the name dlsym and
// GetProcAddress take: ELF dynamic symbols, PE export names, and Mach-O
// external symbols without the leading underscore the platform adds.
func exports(path string) (map[string]bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	var magic [4]byte
	if _, err := f.ReadAt(magic[:], 0); err != nil {
		return nil, err
	}
	switch {
	case bytes.Equal(magic[:], []byte(elf.ELFMAG)):
		return elfExports(path)
	case magic[0] == 'M' && magic[1] == 'Z':
		return peExports(path)
	default:
		return machoExports(path)
	}
}

func elfExports(path string) (map[string]bool, error) {
	f, err := elf.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	syms, err := f.DynamicSymbols()
	if err != nil {
		return nil, err
	}
	out := map[string]bool{}
	for _, s := range syms {
		if s.Section != elf.SHN_UNDEF && (elf.ST_BIND(s.Info) == elf.STB_GLOBAL || elf.ST_BIND(s.Info) == elf.STB_WEAK) {
			out[s.Name] = true
		}
	}
	return out, nil
}

func machoExports(path string) (map[string]bool, error) {
	var f *macho.File
	if fat, err := macho.OpenFat(path); err == nil {
		defer func() { _ = fat.Close() }()
		if len(fat.Arches) == 0 {
			return nil, errors.New("empty universal binary")
		}
		f = fat.Arches[0].File
	} else {
		f, err = macho.Open(path)
		if err != nil {
			return nil, err
		}
		defer func() { _ = f.Close() }()
	}
	if f.Symtab == nil {
		return nil, errors.New("no symbol table")
	}
	const nExt, nTypeMask, nSect = 0x01, 0x0e, 0x0e
	out := map[string]bool{}
	for _, s := range f.Symtab.Syms {
		if s.Type&nExt != 0 && s.Type&nTypeMask == nSect && len(s.Name) > 1 && s.Name[0] == '_' {
			out[s.Name[1:]] = true
		}
	}
	return out, nil
}

// peExports reads the export directory, which debug/pe does not parse.
func peExports(path string) (map[string]bool, error) {
	f, err := pe.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	var dir pe.DataDirectory
	switch oh := f.OptionalHeader.(type) {
	case *pe.OptionalHeader64:
		dir = oh.DataDirectory[pe.IMAGE_DIRECTORY_ENTRY_EXPORT]
	case *pe.OptionalHeader32:
		dir = oh.DataDirectory[pe.IMAGE_DIRECTORY_ENTRY_EXPORT]
	default:
		return nil, errors.New("no optional header")
	}
	if dir.Size == 0 {
		return map[string]bool{}, nil
	}
	// at returns the bytes of the section holding rva, from rva on.
	cache := map[*pe.Section][]byte{}
	at := func(rva uint32) ([]byte, error) {
		for _, s := range f.Sections {
			if rva >= s.VirtualAddress && rva < s.VirtualAddress+max(s.VirtualSize, s.Size) {
				data, ok := cache[s]
				if !ok {
					var err error
					if data, err = s.Data(); err != nil {
						return nil, err
					}
					cache[s] = data
				}
				off := int(rva - s.VirtualAddress)
				if off >= len(data) {
					return nil, fmt.Errorf("rva %#x is past the data of %s", rva, s.Name)
				}
				return data[off:], nil
			}
		}
		return nil, fmt.Errorf("rva %#x is in no section", rva)
	}
	read := func(rva, n uint32) ([]byte, error) {
		b, err := at(rva)
		if err != nil {
			return nil, err
		}
		if uint32(len(b)) < n {
			return nil, fmt.Errorf("rva %#x: short read", rva)
		}
		return b[:n], nil
	}
	hdr, err := read(dir.VirtualAddress, 40)
	if err != nil {
		return nil, err
	}
	le := binary.LittleEndian
	numNames := le.Uint32(hdr[24:])
	namesRVA := le.Uint32(hdr[32:])
	table, err := read(namesRVA, 4*numNames)
	if err != nil {
		return nil, err
	}
	out := map[string]bool{}
	for i := uint32(0); i < numNames; i++ {
		rva := le.Uint32(table[4*i:])
		b, err := at(rva)
		if err != nil {
			return nil, err
		}
		end := bytes.IndexByte(b, 0)
		if end < 0 {
			return nil, fmt.Errorf("export name at %#x is not terminated", rva)
		}
		name := b[:end]
		out[string(name)] = true
	}
	return out, nil
}

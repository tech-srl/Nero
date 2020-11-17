

import logging
import magic

from subprocess import getstatusoutput
from os.path import basename, join as path_join, islink
from elftools.elf.elffile import ELFFile


# readelf -d gcc-5__Ou__pcp-4.0.0__PMDA.so | grep 'NEEDED'
from scandir import walk


def fast_procedure_exists_in_elf(elf_path, procedure_name):
    # clang and icc put the dwarf in a different order, so we need 10 lines (just to be safe)
    cmd = "objdump --dwarf=info {}  | grep DW_TAG_subprogram -A 10 | grep DW_AT_name  | grep {}".format(elf_path,
                                                                                                        procedure_name)
    out = getstatusoutput(cmd)
    return out[0] == 0


# libname should already be with .so if applicable
def fast_elf_uses_lib(elf_path, lib_name):
    # clang and icc put the dwarf in a different order, so we need 10 lines (just to be safe)
    cmd = "readelf -d {} | grep \"Shared library:\" | grep {} ".format(elf_path, lib_name)
    out = getstatusoutput(cmd)
    return out[0] == 0


def fast_elf_libs(elf_path):
    # | grep \"Shared library:\"
    dynamic = getstatusoutput("readelf -d {} | grep \"Shared library:\"".format(elf_path))

    if dynamic[0] == 0:
        arch = getstatusoutput("readelf -h {} | grep Class".format(elf_path))
        assert (arch[0] == 0)

        arch = arch[1].split("Class:")[1].strip()

        def get_lib_name(line):

            splitted = line.split("Shared library: ")
            if len(splitted) != 2:
                return None
            else:
                return splitted[1][1:][:-1]

        try:
            lib_full_names = [x for x in map(get_lib_name, dynamic[1].split("\n")) if x is not None]
        except IndexError:
            print(dynamic)
        lib_names = [x.split(".so")[0] for x in lib_full_names]
        return lib_names, arch

    return None, None


def procedure_exists_in_elf(elf_path, procedure_name):
    with open(elf_path, 'rb') as f:
        elffile = ELFFile(f)

        if not elffile.has_dwarf_info():
            logging.warn("Ignoring ELF without DWARF info ({})".format(elf_path))
            return False

        dwarfinfo = elffile.get_dwarf_info()
        for cu in dwarfinfo.iter_CUs():
            for die in cu.iter_DIEs():
                if die.tag == "DW_TAG_subprogram":
                    if 'DW_AT_name' in die.attributes:
                        if die.attributes['DW_AT_name'].value == procedure_name:
                            return True

    return False


def get_elf_arch(path):
    import cle
    from datagen.vex2llvm import unmodeled_vex_instructions
    elf_magic = magic.from_file(path)
    if 'corrupted' in elf_magic:
        logging.warning('Skipping corrupted header file {} ({}). Enough with the corruption, power to the people!'
                     .format(basename(path), elf_magic))
        raise CorruptedElfHeader()

    try:
        reader = ELFFile(open(path,"rb"))
        elf_arch = cle.backends.elf.ELF.extract_arch(reader)
    except AttributeError as e:
        logging.warning("Skipping elf with attribute error for file={} (err={})".format(path, e))
        raise UnsupportedArch()

    if elf_arch.bits == 32 and '64' in elf_magic:
        logging.warning('Encountered funky file {}, has both 32 and 64 stuff in it ({}).'
                     .format(basename(path), elf_magic))
        raise CorruptedElfHeader()
    if type(elf_arch) not in list(unmodeled_vex_instructions.Call.Archs.keys()):
        logging.debug("Skipping unknown arch {} for {}".format(elf_arch, path))
        raise UnsupportedArch()

    return elf_arch


class UnsupportedArch(Exception):
    pass


class CorruptedElfHeader(Exception):
    pass


if __name__ == '__main__':
    # http://ftp.gnu.org/gnu/wget/wget-1.15.tar.xz complied with -gdwarf

    # CVE-2014-2855
    # https://download.samba.org/pub/rsync/src/rsync-3.1.0.tar.gz
    # https://download.samba.org/pub/rsync/src/rsync-3.1.1.tar.gz

    # CVE-2014-3468
    # http://ftp.gnu.org/gnu/libtasn1/libtasn1-3.6.tar.gz

    assert (fast_procedure_exists_in_elf("wget", 'ftp_retrieve_glob', True))
    assert (not fast_procedure_exists_in_elf("wget", 'ftp_retrieve_glob2', True))
    assert (fast_procedure_exists_in_elf("rsync", 'check_secret', True))

    print("Tests OK")

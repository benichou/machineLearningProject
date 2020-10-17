from Bio import SeqIO

with open("C:\\Users\\franc\\Downloads\\BWRF_03_27_2020.fastq", "rU") as handle:
    for record in SeqIO.parse(handle, "fastq"):
        print(record.id)



first_record = next(SeqIO.parse("C:\\Users\\franc\\Downloads\\BWRF_03_27_2020.fastq", "fastq"))

print("hello world")
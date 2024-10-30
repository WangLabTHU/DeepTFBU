def if_overlap(intervals,chrom,start,end):
    overlap = False
    if chrom in intervals:
        for interval in intervals[chrom]:
            if (start>end)|(interval[0]>interval[1]):
                raise Exception("error in bed")
            if max(start, interval[0]) < min(end, interval[1]):
                overlap = True
                break
    if overlap:
        return(1)
    else:
        return(0)


if __name__ == '__main__':
    histone_intervals_me3 = {}
    histone_bed = './data_dir/HepG2_H3K4me3.bed' # H3K4me3 histome modification bed narrow peak file
    with open(histone_bed, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if chrom not in histone_intervals_me3:
                histone_intervals_me3[chrom] = []
            histone_intervals_me3[chrom].append((start, end))

    histone_intervals_me1 = {}
    histone_bed = './data_dir/HepG2_H3K4me1.bed' # H3K4me1 histome modification bed narrow peak file
    with open(histone_bed, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if chrom not in histone_intervals_me1:
                histone_intervals_me1[chrom] = []
            histone_intervals_me1[chrom].append((start, end))


    TF_list = ['GATA2']
    for TF_name in TF_list:
        target_bed_pos = './data_dir/step2_HepG2_pos_'+TF_name+'_data.bed'
        target_bed_pos_out = './data_dir/step3_HepG2_pos_'+TF_name+'_data.bed'
        with open(target_bed_pos, 'r') as f_in, open(target_bed_pos_out, 'w') as f_out:
            for line in f_in:
                fields = line.strip().split('\t')
                chrom, start, end = fields[0], int(fields[1]), int(fields[2])
                overlap_me3 = if_overlap(histone_intervals_me3,chrom,start,end)
                overlap_me1 = if_overlap(histone_intervals_me1,chrom,start,end)
                for item in fields:
                    f_out.write(item + '\t')
                f_out.write(str(overlap_me3) + '\t')
                f_out.write(str(overlap_me1) + '\t')
                f_out.write('\n')


        target_bed_neg = './data_dir/step2_HepG2_neg_'+TF_name+'_data.bed'
        target_bed_neg_out = './data_dir/step3_HepG2_neg_'+TF_name+'_data.bed'
        with open(target_bed_neg, 'r') as f_in, open(target_bed_neg_out, 'w') as f_out:
            for line in f_in:
                fields = line.strip().split('\t')
                chrom, start, end = fields[0], int(fields[1]), int(fields[2])
                overlap_me3 = if_overlap(histone_intervals_me3,chrom,start,end)
                overlap_me1 = if_overlap(histone_intervals_me1,chrom,start,end)
                for item in fields:
                    f_out.write(item + '\t')
                f_out.write(str(overlap_me3) + '\t')
                f_out.write(str(overlap_me1) + '\t')
                f_out.write('\n')

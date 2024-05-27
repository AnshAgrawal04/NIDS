from scapy.all import rdpcap, wrpcap
from datetime import datetime
import math
import numpy as np
import os
def merge_pcaps(pcap1, pcap2, output_file,labels_path):
    # Read the PCAP files
    packets1 = rdpcap(pcap1)
    packets2 = rdpcap(pcap2)
    print("whoo")
    # Get the floor of the arrival time for the second file
    floor_time2 = min([pkt.time for pkt in packets2])
    floor_time2 = math.floor(floor_time2)
    print("whee")
    # Calculate the offset for the first file
    time_offset = floor_time2 - math.floor(min([pkt.time for pkt in packets1]))

    # Modify the arrival time for the first file
    for pkt in packets1:
        print(pkt.time)
        pkt.time += time_offset
        print(pkt.time)

    # Merge the packets based on arrival time
    # merged_packets = sorted(packets1 + packets2, key=lambda pkt: pkt.time)
    # labels = np.array([0 if pkt in packets1 else 1 for pkt in merged_packets])
    # np.save("labels.npy", labels)

    # # Write the merged packets to the output file
    # wrpcap(output_file, merged_packets)
    packet1=[[pkt.time,0,pkt] for pkt in packets1]
    packet2=[[pkt.time,1,pkt] for pkt in packets2]
    merged_packets = sorted(packet1 + packet2, key=lambda pkt: pkt[0])
    labels = np.array([pkt[1] for pkt in merged_packets])
    np.save(labels_path, labels)
    merged_packets = [pkt[2] for pkt in merged_packets]
    wrpcap(output_file, merged_packets)

if __name__ == "__main__":
    pcap1 = "weekday.pcap"
    #do for all the files in the directory adv
    for filename in os.listdir('mal'):
        print(filename)
        pcap2 = f"mal/{filename}"
        filename = filename.split(".")[0]
        os.mkdir(f"Mal_{filename}")
        output_file = f"Mal_{filename}/merged_{filename}.pcap"
        labels_path = f"Mal_{filename}/labels.npy"
        print("here")
        merge_pcaps(pcap1, pcap2, output_file,labels_path)

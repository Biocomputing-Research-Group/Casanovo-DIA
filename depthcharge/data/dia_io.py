#from deepnovo_cython_modules import process_spectrum, get_candidate_intensity
import deepnovo_config
import os
import pickle
import re
import numpy as np

#import Cython


class WorkerDIA(object):
    def __init__(self):
        self.MZ_MAX = deepnovo_config.MZ_MAX
        self.MZ_SIZE = deepnovo_config.MZ_SIZE
        self.neighbor_size = deepnovo_config.neighbor_size
        self.mass_H = deepnovo_config.mass_H
        self.dia_window = deepnovo_config.dia_window
        self._buckets =deepnovo_config._buckets
        self.vocab = deepnovo_config.vocab
        self.GO_ID =deepnovo_config.GO_ID
        self.EOS_ID = deepnovo_config.EOS_ID
        self.PAD_ID = deepnovo_config.PAD_ID
        self.vocab_size = deepnovo_config.vocab_size
        self.WINDOW_SIZE = deepnovo_config.WINDOW_SIZE
        self.SPECTRUM_RESOLUTION = deepnovo_config.SPECTRUM_RESOLUTION
        self.mass_ID_np = deepnovo_config.mass_ID_np
        self.mass_H2O = deepnovo_config.mass_H2O
        self.mass_NH3 = deepnovo_config.mass_NH3
        self.num_ion = deepnovo_config.num_ion

    def process_spectrum(self, spectrum_mz_list, spectrum_intensity_list, peptide_mass):


        # neutral mass, location, assuming ion charge z=1
        charge = 1.0
        spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)
        neutral_mass = spectrum_mz - charge * deepnovo_config.mass_H
        neutral_mass_location = np.rint(neutral_mass * deepnovo_config.SPECTRUM_RESOLUTION).astype(
            np.int32)  # TODO(nh2tran): line-too-long
     
        neutral_mass_location_view = neutral_mass_location

        # intensity
        spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
        spectrum_intensity_max = np.max(spectrum_intensity)
        norm_intensity = spectrum_intensity
        norm_intensity_view = norm_intensity

        # fill spectrum holders
        spectrum_holder = np.zeros(shape=(1, deepnovo_config.MZ_SIZE), dtype=np.float32)

        spectrum_holder_view = spectrum_holder

        for index in range(neutral_mass_location.size):
            spectrum_holder_view[0, neutral_mass_location_view[index]] = max(
                spectrum_holder_view[0, neutral_mass_location_view[index]],  
                norm_intensity_view[index])  
        spectrum_original_forward = np.copy(spectrum_holder)
        spectrum_original_backward = np.copy(spectrum_holder)


        # peptide_mass
        spectrum_original_forward[0, int(round(
            peptide_mass * deepnovo_config.SPECTRUM_RESOLUTION))] = spectrum_intensity_max
        spectrum_original_backward[0, int(round(
            peptide_mass * deepnovo_config.SPECTRUM_RESOLUTION))] = spectrum_intensity_max  
        # N-terminal, b-ion, peptide_mass_C
        # append N-terminal
        
        # append peptide_mass_C
        mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
        peptide_mass_C = peptide_mass - mass_C
        spectrum_original_forward[0, int(round(
            peptide_mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = spectrum_intensity_max  
        # C-terminal, y-ion, peptide_mass_N
        
        # append peptide_mass_N
        mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
        peptide_mass_N = peptide_mass - mass_N
        spectrum_original_backward[0, int(round(
            peptide_mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = spectrum_intensity_max

        return spectrum_original_forward, spectrum_original_backward

    def get_location(self, peptide_mass, prefix_mass, direction):
        if direction == 0:
            candidate_b_mass = prefix_mass + self.mass_ID_np
            candidate_y_mass = peptide_mass - candidate_b_mass
        elif direction == 1:
            candidate_y_mass = prefix_mass + self.mass_ID_np
            candidate_b_mass = peptide_mass - candidate_y_mass

        # b-ions
        candidate_b_H2O = candidate_b_mass - self.mass_H2O
        candidate_b_NH3 = candidate_b_mass - self.mass_NH3
        candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * self.mass_H) / 2
                                     - self.mass_H)

        # y-ions
        candidate_y_H2O = candidate_y_mass - self.mass_H2O
        candidate_y_NH3 = candidate_y_mass - self.mass_NH3
        candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * self.mass_H) / 2
                                     - self.mass_H)

        # ion_2
        # ~   b_ions = [candidate_b_mass]
        # ~   y_ions = [candidate_y_mass]
        # ~   ion_mass_list = b_ions + y_ions

        # ion_8
        b_ions = [candidate_b_mass,
                  candidate_b_H2O,
                  candidate_b_NH3,
                  candidate_b_plus2_charge1]
        y_ions = [candidate_y_mass,
                  candidate_y_H2O,
                  candidate_y_NH3,
                  candidate_y_plus2_charge1]
        ion_mass_list = b_ions + y_ions
        ion_mass = np.array(ion_mass_list, dtype=np.float32)

        # ion locations
        location_sub50 = np.rint(ion_mass * self.SPECTRUM_RESOLUTION).astype(np.int32)
        location_sub50 -= (self.WINDOW_SIZE // 2)
        location_plus50 = location_sub50 + self.WINDOW_SIZE
        ion_id_rows, aa_id_cols = np.nonzero(np.logical_and(
            location_sub50 >= 0,
            location_plus50 <= self.MZ_SIZE))
        return ion_id_rows, aa_id_cols, location_sub50, location_plus50

    
    def copy_values(self, candidate_intensity_view, spectrum_view, location_sub,i1, i2):
    
        i1_start = self.neighbor_size * i1
        for neighbor in range(self.neighbor_size):
            for j in range(self.WINDOW_SIZE):
                try:
                    candidate_intensity_view[i2, i1_start + neighbor, j] = spectrum_view[neighbor, location_sub[i1, i2] + j]
                except:
                    continue
                    
    def get_candidate_intensity(self, spectrum_original, peptide_mass, prefix_mass, direction):
        ion_id_rows, aa_id_cols, location_sub50, location_plus50 = self.get_location(peptide_mass, prefix_mass, direction)
        # candidate_intensity
        candidate_intensity = np.zeros(shape=(self.vocab_size,
                                          self.neighbor_size * self.num_ion,
                                          self.WINDOW_SIZE),
                                   dtype=np.float32)
   
        location_sub50_view = location_sub50
 
        location_plus50_view = location_plus50
        candidate_intensity_view = candidate_intensity
        row = ion_id_rows.astype(np.int32)
        col = aa_id_cols.astype(np.int32)
        
        for index in range(ion_id_rows.size):
            if col[index] < 3:
                continue
            self.copy_values(candidate_intensity_view, spectrum_original, location_sub50_view, row[index], col[index])
        max_intensity = np.max(candidate_intensity)
        if max_intensity > 1.0:
            candidate_intensity /= max_intensity
        return candidate_intensity

    def _parse_spectrum(self, precursor_mass, scan_list_middle, mz_lists, intensity_lists, neighbor_right_count, neighbor_size_half):
        
        MZ_SIZE = int(self.MZ_MAX * self.SPECTRUM_RESOLUTION)
        spectrum_original_forward_list = []
        spectrum_original_backward_list = []
        for mz_list, intensity_list in zip(mz_lists, intensity_lists):

            spectrum_original_forward, spectrum_original_backward = self.process_spectrum(mz_list,
                                                                 intensity_list,
                                                                 precursor_mass)
            spectrum_original_forward_list.append(spectrum_original_forward)
            spectrum_original_backward_list.append(spectrum_original_backward)

        if neighbor_right_count < neighbor_size_half:
            for x in range(neighbor_size_half - neighbor_right_count):
                spectrum_original_forward_list.append(np.zeros(
                    shape=(1, deepnovo_config.MZ_SIZE),
                    dtype=np.float32))
                spectrum_original_backward_list.append(np.zeros(
                    shape=(1, self.MZ_SIZE),
                    dtype=np.float32))

        spectrum_original_forward = np.vstack(spectrum_original_forward_list)
        spectrum_original_backward = np.vstack(spectrum_original_backward_list)
       
        return spectrum_original_forward, spectrum_original_backward

    def _process_raw_seq(self, raw_seq):
        peptide = []
        raw_len = len(raw_seq)
        pep_len, index = 0, 0
        while index < raw_len:
            if raw_seq[index] == '+':
                if peptide[-1] == 'C' and raw_seq[index:index + 7] == '+57.021':
                    peptide[-1] = 'C(Carbamidomethylation)'
                    index += 7
            else:
                peptide.append(raw_seq[index])
                index += 1
                pep_len += 1
        return peptide, pep_len

    def calculate_ms2(self, precursor_mz, precursor_charge, scan_list_middle, mz_list, int_list, neighbor_right_count, neighbor_size_half, seq):
        candidate_intensity_list_forward = []
        candidate_intensity_list_backward = []

        precursor_mass = precursor_mz * precursor_charge - self.mass_H * precursor_charge

        if precursor_mass > self.MZ_MAX:
            return None
        spectrum_original_forward, spectrum_original_backward = self._parse_spectrum(precursor_mass, scan_list_middle, mz_list, int_list, neighbor_right_count, neighbor_size_half)
        peptide, peptide_len = self._process_raw_seq(seq)
        for bucket_id, target_size in enumerate(self._buckets):
            if peptide_len + 2 <= target_size:  # +2 to include GO and EOS
                break
        decoder_size = self._buckets[bucket_id]
        peptide_ids = [self.vocab[x] for x in peptide]
        pad_size = decoder_size - (len(peptide_ids) + 2)
        # forward

        peptide_ids_forward = peptide_ids[:]
        peptide_ids_forward.insert(0, self.GO_ID)
        peptide_ids_forward.append(self.EOS_ID)
        peptide_ids_forward += [self.PAD_ID] * pad_size
        
        peptide_ids_backward = peptide_ids[::-1]
        peptide_ids_backward.insert(0, deepnovo_config.EOS_ID)
        peptide_ids_backward.append(deepnovo_config.GO_ID)
        peptide_ids_backward += [deepnovo_config.PAD_ID] * pad_size

        prefix_mass = 0.0
        suffix_mass = 0.0
        for index in range(self._buckets[-1]):
            if index < decoder_size:
                prefix_mass += deepnovo_config.mass_ID[peptide_ids_forward[index]]
                candidate_intensity_forward = self.get_candidate_intensity(
                    spectrum_original_forward,
                    precursor_mass,
                    prefix_mass,
                    0)
                cand_int_shape_forward = candidate_intensity_forward.shape

                suffix_mass += deepnovo_config.mass_ID[peptide_ids_backward[index]]
                candidate_intensity_backward = self.get_candidate_intensity(
                    spectrum_original_backward,
                    precursor_mass,
                    suffix_mass,
                    1)
                cand_int_shape_backward = candidate_intensity_backward.shape
            else:
                candidate_intensity_forward = np.zeros(cand_int_shape_forward)
                candidate_intensity_backward = np.zeros(cand_int_shape_backward)

            candidate_intensity_list_forward.append(candidate_intensity_forward)
            candidate_intensity_list_backward.append(candidate_intensity_backward)
        return candidate_intensity_list_forward, candidate_intensity_list_backward

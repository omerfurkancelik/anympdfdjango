import re
import os
import uuid
import PyPDF2
import nltk
import spacy
from django.conf import settings
from django.core.files.base import ContentFile
import logging
import fitz  # PyMuPDF
import io
from PIL import Image, ImageDraw, ImageFilter
import tempfile
import numpy as np
import cv2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize logging
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("Could not load spaCy model. Running spaCy download...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    
    
    
import os
import base64
import hashlib
import logging
import binascii

logger = logging.getLogger(__name__)

class AES:
    """
    Pure Python implementation of AES (Advanced Encryption Standard)
    Implements AES-128, AES-192, and AES-256
    """
    
    # S-box for SubBytes operation
    sbox = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    ]
    
    # Inverse S-box for InvSubBytes operation
    inv_sbox = [
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
    ]
    
    # Round constants for key expansion
    rcon = [
        0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a
    ]
    
    def __init__(self, key):
        """
        Initialize AES with the given key
        
        Args:
            key (bytes): The encryption key (16, 24, or 32 bytes for AES-128, AES-192, or AES-256)
        """
        self.key_size = len(key)
        
        # Ensure key size is valid
        if self.key_size not in [16, 24, 32]:
            raise ValueError("Key size must be 16, 24, or 32 bytes (128, 192, or 256 bits)")
        
        # Determine number of rounds based on key size
        self.rounds = {16: 10, 24: 12, 32: 14}[self.key_size]
        
        # Number of 32-bit words in the key
        self.Nk = self.key_size // 4
        
        # Size of state matrix in 32-bit words (always 4 for AES)
        self.Nb = 4
        
        # Generate round keys
        self.key_schedule = self.key_expansion(key)
    
    def sub_word(self, word):
        """
        Apply S-box substitution to each byte of a word
        
        Args:
            word (list): A list of 4 bytes
            
        Returns:
            list: Substituted word
        """
        return [self.sbox[b] for b in word]
    
    def rot_word(self, word):
        """
        Rotate left: [a0, a1, a2, a3] -> [a1, a2, a3, a0]
        
        Args:
            word (list): A list of 4 bytes
            
        Returns:
            list: Rotated word
        """
        return word[1:] + word[:1]
    
    def key_expansion(self, key):
        """
        Expand the key into a key schedule
        
        Args:
            key (bytes): The encryption key
            
        Returns:
            list: The expanded key schedule
        """
        # Convert key to list of bytes
        key_bytes = list(key)
        
        # Total number of words in expanded key
        total_words = 4 * (self.rounds + 1)
        
        # Initialize expanded key with empty list
        w = [0] * total_words
        
        # Copy original key bytes to first Nk words
        for i in range(self.Nk):
            w[i] = key_bytes[4*i:4*i+4]
        
        # Generate the rest of the expanded key
        for i in range(self.Nk, total_words):
            temp = list(w[i-1])  # Make a copy
            
            if i % self.Nk == 0:
                temp = self.sub_word(self.rot_word(temp))
                temp[0] ^= self.rcon[i // self.Nk]
            elif self.Nk > 6 and i % self.Nk == 4:
                temp = self.sub_word(temp)
            
            w[i] = [w[i-self.Nk][j] ^ temp[j] for j in range(4)]
        
        return w
    
    def add_round_key(self, state, round_key):
        """
        XOR the state with the round key
        
        Args:
            state (list): 4x4 state matrix
            round_key (list): The round key
            
        Returns:
            list: Updated state matrix
        """
        for i in range(4):
            for j in range(4):
                state[i][j] ^= round_key[i + 4*j]
        return state
    
    def sub_bytes(self, state):
        """
        Apply S-box substitution to each byte in state
        
        Args:
            state (list): 4x4 state matrix
            
        Returns:
            list: Updated state matrix
        """
        for i in range(4):
            for j in range(4):
                state[i][j] = self.sbox[state[i][j]]
        return state
    
    def inv_sub_bytes(self, state):
        """
        Apply inverse S-box substitution to each byte in state
        
        Args:
            state (list): 4x4 state matrix
            
        Returns:
            list: Updated state matrix
        """
        for i in range(4):
            for j in range(4):
                state[i][j] = self.inv_sbox[state[i][j]]
        return state
    
    def shift_rows(self, state):
        """
        Shift rows of state: no shift in row 0, 1-byte shift in row 1, etc.
        
        Args:
            state (list): 4x4 state matrix
            
        Returns:
            list: Updated state matrix
        """
        # Row 0: No shift
        # Row 1: Shift left by 1
        state[1] = state[1][1:] + state[1][:1]
        # Row 2: Shift left by 2
        state[2] = state[2][2:] + state[2][:2]
        # Row 3: Shift left by 3
        state[3] = state[3][3:] + state[3][:3]
        return state
    
    def inv_shift_rows(self, state):
        """
        Inverse shift rows of state for decryption
        
        Args:
            state (list): 4x4 state matrix
            
        Returns:
            list: Updated state matrix
        """
        # Row 0: No shift
        # Row 1: Shift right by 1
        state[1] = state[1][3:] + state[1][:3]
        # Row 2: Shift right by 2
        state[2] = state[2][2:] + state[2][:2]
        # Row 3: Shift right by 3
        state[3] = state[3][1:] + state[3][:1]
        return state
    
    def xtime(self, a):
        """
        Multiply by x (in GF(2^8))
        
        Args:
            a (int): Byte to multiply
            
        Returns:
            int: Result
        """
        if a & 0x80:
            return ((a << 1) ^ 0x1b) & 0xff
        else:
            return (a << 1) & 0xff
    
    def mix_single_column(self, col):
        """
        Mix a single column in the state matrix
        
        Args:
            col (list): A column of 4 bytes
            
        Returns:
            list: Mixed column
        """
        t = col[0] ^ col[1] ^ col[2] ^ col[3]
        u = col[0]
        col[0] ^= t ^ self.xtime(col[0] ^ col[1])
        col[1] ^= t ^ self.xtime(col[1] ^ col[2])
        col[2] ^= t ^ self.xtime(col[2] ^ col[3])
        col[3] ^= t ^ self.xtime(col[3] ^ u)
        return col
    
    def mix_columns(self, state):
        """
        Mix all columns in the state matrix
        
        Args:
            state (list): 4x4 state matrix
            
        Returns:
            list: Updated state matrix
        """
        for i in range(4):
            column = [state[j][i] for j in range(4)]
            column = self.mix_single_column(column)
            for j in range(4):
                state[j][i] = column[j]
        return state
    
    def inv_mix_columns(self, state):
        """
        Inverse mix columns for decryption
        
        Args:
            state (list): 4x4 state matrix
            
        Returns:
            list: Updated state matrix
        """
        for i in range(4):
            column = [state[j][i] for j in range(4)]
            # Implementation of multiplication in GF(2^8)
            a = [0] * 4
            b = [0] * 4
            
            for j in range(4):
                a[j] = column[j]
                b[j] = column[j]
                if (j + 1) % 4 < 2:
                    b[j] = self.xtime(b[j])
                if (j + 1) % 4 == 1:
                    a[j] = self.xtime(a[j])
            
            column[0] = self.xtime(self.xtime(a[0] ^ a[2])) ^ a[0] ^ b[1] ^ a[2] ^ a[3]
            column[1] = self.xtime(self.xtime(a[1] ^ a[3])) ^ a[1] ^ b[2] ^ a[0] ^ a[3]
            column[2] = self.xtime(self.xtime(a[2] ^ a[0])) ^ a[2] ^ b[3] ^ a[0] ^ a[1]
            column[3] = self.xtime(self.xtime(a[3] ^ a[1])) ^ a[3] ^ b[0] ^ a[1] ^ a[2]
            
            for j in range(4):
                state[j][i] = column[j]
        
        return state
    
    def bytes_to_state(self, data):
        """
        Convert a 16-byte array to 4x4 state matrix
        
        Args:
            data (bytes): 16 bytes of data
            
        Returns:
            list: 4x4 state matrix
        """
        state = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                state[i][j] = data[i + 4*j]
        return state
    
    def state_to_bytes(self, state):
        """
        Convert 4x4 state matrix to 16-byte array
        
        Args:
            state (list): 4x4 state matrix
            
        Returns:
            bytes: 16 bytes of data
        """
        output = bytearray(16)
        for i in range(4):
            for j in range(4):
                output[i + 4*j] = state[i][j]
        return bytes(output)
    
    def encrypt_block(self, plaintext):
        """
        Encrypt a single 16-byte block
        
        Args:
            plaintext (bytes): 16 bytes of plaintext
            
        Returns:
            bytes: 16 bytes of ciphertext
        """
        if len(plaintext) != 16:
            raise ValueError("Plaintext block must be 16 bytes")
        
        # Convert plaintext to state matrix
        state = self.bytes_to_state(plaintext)
        
        # Initial round key addition
        state = self.add_round_key(state, self.key_schedule[0:4])
        
        # Main rounds
        for round_num in range(1, self.rounds):
            state = self.sub_bytes(state)
            state = self.shift_rows(state)
            state = self.mix_columns(state)
            state = self.add_round_key(state, self.key_schedule[4*round_num:4*(round_num+1)])
        
        # Final round (no mix columns)
        state = self.sub_bytes(state)
        state = self.shift_rows(state)
        state = self.add_round_key(state, self.key_schedule[4*self.rounds:4*(self.rounds+1)])
        
        # Convert state matrix back to bytes
        return self.state_to_bytes(state)
    
    def decrypt_block(self, ciphertext):
        """
        Decrypt a single 16-byte block
        
        Args:
            ciphertext (bytes): 16 bytes of ciphertext
            
        Returns:
            bytes: 16 bytes of plaintext
        """
        if len(ciphertext) != 16:
            raise ValueError("Ciphertext block must be 16 bytes")
        
        # Convert ciphertext to state matrix
        state = self.bytes_to_state(ciphertext)
        
        # Initial round key addition (with last round key)
        state = self.add_round_key(state, self.key_schedule[4*self.rounds:4*(self.rounds+1)])
        
        # Main rounds in reverse
        for round_num in range(self.rounds-1, 0, -1):
            state = self.inv_shift_rows(state)
            state = self.inv_sub_bytes(state)
            state = self.add_round_key(state, self.key_schedule[4*round_num:4*(round_num+1)])
            state = self.inv_mix_columns(state)
        
        # Final round (reverse of initial round)
        state = self.inv_shift_rows(state)
        state = self.inv_sub_bytes(state)
        state = self.add_round_key(state, self.key_schedule[0:4])
        
        # Convert state matrix back to bytes
        return self.state_to_bytes(state)


class AESCipher:
    """
    Complete AES-CBC implementation for PDF anonymization
    """
    
    def __init__(self, key=None):
        """
        Initialize AES cipher with a key
        If no key provided, generate a secure random key
        
        Args:
            key (bytes, optional): AES encryption key
        """
        self.key = key if key else os.urandom(32)  # 256-bit key
        # Hash the key to ensure it's the right size for AES
        self.aes = AES(hashlib.sha256(self.key).digest()[:32])
    
    def pad(self, data):
        """
        PKCS#7 padding
        
        Args:
            data (bytes): Data to pad
            
        Returns:
            bytes: Padded data
        """
        pad_len = 16 - (len(data) % 16)
        return data + bytes([pad_len] * pad_len)
    
    def unpad(self, data):
        """
        Remove PKCS#7 padding
        
        Args:
            data (bytes): Padded data
            
        Returns:
            bytes: Original data
        """
        pad_len = data[-1]
        if pad_len > 16:
            raise ValueError("Invalid padding")
        return data[:-pad_len]
    
    def encrypt(self, text):
        """
        Encrypt text using AES-256 in CBC mode
        
        Args:
            text (str): Text to encrypt
            
        Returns:
            str: Base64 encoded encrypted text with IV
        """
        try:
            # Convert text to bytes if it's not already
            if isinstance(text, str):
                text_bytes = text.encode('utf-8')
            else:
                text_bytes = text
            
            # Generate a random IV (Initialization Vector)
            iv = os.urandom(16)
            
            # Pad the plaintext
            padded_text = self.pad(text_bytes)
            
            # Initialize result with the IV
            result = bytearray(iv)
            
            # Current block for CBC mode
            prev_block = iv
            
            # Process each block
            for i in range(0, len(padded_text), 16):
                block = padded_text[i:i+16]
                
                # XOR with previous ciphertext block (or IV for first block)
                xored_block = bytes(x ^ y for x, y in zip(block, prev_block))
                
                # Encrypt the block
                encrypted_block = self.aes.encrypt_block(xored_block)
                
                # Add to result
                result.extend(encrypted_block)
                
                # Update previous block
                prev_block = encrypted_block
            
            # Encode to base64
            encrypted_package = base64.b64encode(result)
            
            # Return as string
            return encrypted_package.decode('utf-8')
        
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            # Return a placeholder if encryption fails
            return f"[ENC_ERROR_{text}]"
    
    def decrypt(self, encrypted_text):
        """
        Decrypt text encrypted with AES-256 in CBC mode
        
        Args:
            encrypted_text (str): Base64 encoded encrypted text with IV
            
        Returns:
            str: Decrypted text
        """
        try:
            # Decode base64
            encrypted_package = base64.b64decode(encrypted_text)
            
            # Extract IV (first 16 bytes)
            iv = encrypted_package[:16]
            ciphertext = encrypted_package[16:]
            
            # Initialize result
            result = bytearray()
            
            # Previous block for CBC mode
            prev_block = iv
            
            # Process each block
            for i in range(0, len(ciphertext), 16):
                block = ciphertext[i:i+16]
                
                # Decrypt the block
                decrypted_block = self.aes.decrypt_block(block)
                
                # XOR with previous ciphertext block (or IV for first block)
                xored_block = bytes(x ^ y for x, y in zip(decrypted_block, prev_block))
                
                # Add to result
                result.extend(xored_block)
                
                # Update previous block
                prev_block = block
            
            # Unpad the result
            unpadded = self.unpad(result)
            
            # Return as string
            return unpadded.decode('utf-8')
        
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            # Return a placeholder if decryption fails
            return f"[DEC_ERROR_{encrypted_text}]"
    
    def get_key_hex(self):
        """Return the key as a hex string for storage or transmission"""
        return self.key.hex()
    
    @classmethod
    def from_key_hex(cls, key_hex):
        """Create an AESCipher instance from a hex key string"""
        key = bytes.fromhex(key_hex)
        return cls(key)
    
    
    



def create_anonymization_map(authors, institutions):
    """
    Create a mapping of original text to AES encrypted text
    Returns a dictionary {original: encrypted}
    """
    # Create a new AES cipher instance
    cipher = AESCipher()
    
    # Get the key for storage
    encryption_key = cipher.get_key_hex()
    
    anon_map = {}
    
    # Encrypt authors
    for idx, author in enumerate(authors):
        encrypted_author = cipher.encrypt(author)
        anon_map[author] = encrypted_author
    
    # Encrypt institutions
    for idx, institution in enumerate(institutions):
        encrypted_institution = cipher.encrypt(institution)
        anon_map[institution] = encrypted_institution
    
    # Return both the map and the key
    return anon_map, encryption_key


def decrypt_anonymization_map(anon_map, encryption_key):
    """
    Decrypt an anonymization map using the provided key
    Returns a dictionary {encrypted: original}
    """
    # Create AES cipher instance from the key
    cipher = AESCipher.from_key_hex(encryption_key)
    
    # Create a reverse mapping (encrypted to original)
    reverse_map = {}
    
    for original, encrypted in anon_map.items():
        # Store in reverse order for decryption
        reverse_map[encrypted] = original
    
    return reverse_map, cipher
    

def extract_text_from_pdf(pdf_path):
    """
    PDF'den metin okuyan basit bir fonksiyon örneği.
    PyMuPDF (fitz) veya PyPDF2 kullanılabilir.
    """
    text = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""
        
        
    with open("denemetext.txt","w",encoding="utf-8") as f:
        f.write(text)
    return text

def extract_authors(text):
    """
    Metinden potansiyel yazar isimlerini çıkarmaya çalışır.
    1) 'AUTHORS?' veya 'by' gibi kalıpları yakalar.
    2) Bulamazsa spaCy ile PERSON etiketlerini çeker.
    3) Doktora vb. unvanları siler, gereksiz kısımları temizler.
    """
    # Regex ile denenecek kalıplar
    author_section_patterns = [
        r'(?i)AUTHORS?:?\s*(.*?)(?:\n\n|\n\s*ABSTRACT)',
        r'(?i)(?:by|written by)\s+(.*?)(?:\n\n|\n\s*[A-Z]{2,})',
    ]
    
    authors = []
    found_via_regex = False
    for pattern in author_section_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            found_via_regex = True
            author_text = match.group(1)
            # virgül, "and", ";" vb. ayır
            potential_authors = re.split(r',|\band\b|;', author_text)
            authors.extend([a.strip() for a in potential_authors if a.strip()])
    
    # Eğer Regex yoluyla bulamadıysak spaCy PERSON etiketlerini kullanalım
    if not found_via_regex:
        # Çok büyük metinleri kısaltabiliriz (örnek: ilk 5000 karakter)
        doc = nlp(text[:5000])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                authors.append(ent.text.strip())
    
    # Unvanları temizle (Dr., Prof. vb.)
    cleaned_authors = []
    for a in authors:
        clean_a = re.sub(r'\b(Dr|Prof|PhD|MD|MSc|BSc|BA|MA)\.?', '', a, flags=re.IGNORECASE)
        # Birden fazla boşluğu teke indir
        clean_a = re.sub(r'\s+', ' ', clean_a).strip()
        # Çok uzun olmayan (örn. en fazla 4 kelime) isimleri al
        if clean_a and len(clean_a.split()) <= 4:
            cleaned_authors.append(clean_a)
    
    # Tekrarlıları temizleyelim
    cleaned_authors = list(set(cleaned_authors))
    logger.info(f"[extract_authors] Found authors: {cleaned_authors}")
    return cleaned_authors

def extract_institutions(text):
    """
    Metinden potansiyel kurum (affiliation) bilgilerini çıkarır.
    1) Regex ile 'affiliation', 'department' vb. kısımlara bakar.
    2) Yoksa spaCy ile ORG etiketlerini çeker.
    3) İçinde 'university', 'institute' vb. geçenleri tercih eder.
    """
    # Denenecek kalıplar (affiliation, institution vs.)
    affiliation_patterns = [
        r'(?i)(affiliation|department|institution)s?:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})',
        r'(?i)(?:\n)(?:\d\s+)?(affiliation|department|institution)s?:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})'
    ]
    institutions = []
    
    found_via_regex = False
    for pattern in affiliation_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            found_via_regex = True
            # group(2) varsa oradan alalım, zira group(1) 'affiliation' kelimesinin kendisi olabilir
            affiliation_text = match.group(2) if match.lastindex == 2 else match.group(1)
            potential_insts = re.split(r';|\n|,', affiliation_text)
            institutions.extend([i.strip() for i in potential_insts if i.strip()])
    
    if not found_via_regex:
        # spaCy ile ORG yakalayalım
        doc = nlp(text[:8000])
        # Basit bir tespit kriteri: 'university', 'institute', vs. varsa al
        key_words = ['university','universidade','institute','college','school','department','lab','laboratory',
                     'foundation','company','inc','co.','corp','centre','center']
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # metin küçük harfe dönüştürüp anahtar kelimeler var mı bak
                lower_org = ent.text.lower()
                if any(kw in lower_org for kw in key_words):
                    institutions.append(ent.text.strip())
    
    # Fazla boşlukları temizle
    cleaned_inst = []
    for inst in institutions:
        c = re.sub(r'\s+', ' ', inst).strip()
        if c:
            cleaned_inst.append(c)
    
    cleaned_inst = list(set(cleaned_inst))
    logger.info(f"[extract_institutions] Found institutions: {cleaned_inst}")
    return cleaned_inst

def extract_keywords(text, min_keywords=3, max_keywords=10):
    """
    Metinden (Keywords, Index Terms) bölümlerini veya yoksa
    abstract üzerinden anahtar kelimeleri bulur.
    Burada örneğin spaCy + basit regex yaklaşımı.
    """
    # 1) INDEX TERMS / KEYWORDS bölümünü ara
    keyword_section_patterns = [
        r'(?i)(?:index\s*terms|keywords):?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})',
    ]
    
    explicit_keywords = []
    for pattern in keyword_section_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            kw_text = match.group(1)
            # virgülle böl
            potential_kws = re.split(r',|;', kw_text)
            explicit_keywords.extend([k.strip() for k in potential_kws if k.strip()])
    
    if explicit_keywords:
        # Bulunan keywordler
        unique_kws = list(set(explicit_keywords))
        return unique_kws[:max_keywords]
    
    # 2) Eğer bulamadıysak, Abstract üzerinden spaCy noun_chunks / named_entities
    abstract_pattern = r'(?i)abstract:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})'
    match_abstract = re.search(abstract_pattern, text, flags=re.DOTALL)
    if match_abstract:
        abstract_text = match_abstract.group(1)
    else:
        # Abstract yoksa metnin ilk ~3000 karakterini kullanıyoruz
        abstract_text = text[:3000]
    
    doc = nlp(abstract_text)
    
    # Noun chunks + ORG, PRODUCT vb. entity
    candidates = []
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:  # fazla uzun olmayan ifadeler
            candidates.append(chunk.text.lower())
    
    for ent in doc.ents:
        if ent.label_ not in ("PERSON", "CARDINAL", "DATE", "TIME"):
            candidates.append(ent.text.lower())
    
    # Bir sıklık hesaplaması
    from collections import Counter
    kw_counter = Counter(candidates)
    most_common = [x for x, _ in kw_counter.most_common(max_keywords*2)]
    
    # min_keywords kadar geri döndür
    final_kws = []
    for mc in most_common:
        if mc not in final_kws:
            final_kws.append(mc)
        if len(final_kws) >= max_keywords:
            break
    
    if len(final_kws) < min_keywords:
        # fallback: en azından min_keywords döndür
        return final_kws + ["keyword"]*(min_keywords - len(final_kws))
    
    logger.info(f"[extract_keywords] Found keywords: {final_kws}")
    return final_kws

def anonymize_pdf(pdf_path, anonymize_map):
    """
    Create an anonymized version of the PDF where author and institution names
    are replaced with their AES encrypted versions
    
    Args:
        pdf_path (str): Path to the original PDF
        anonymize_map (dict): Dictionary mapping original text to encrypted text
    
    Returns:
        str: Path to the anonymized PDF
    """
    try:
        # Create a temporary file for the anonymized PDF
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_fd)
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Process each page
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            
            # 1. Text anonymization - this is the most reliable part
            for original, encrypted in anonymize_map.items():
                # Search for text instances and replace them
                text_instances = page.search_for(original)
                for inst in text_instances:
                    # Create a redaction annotation for this text - use the encrypted version
                    anon_text = encrypted
                    # Make sure it's a reasonable length to display in the PDF
                    if len(anon_text) > 50:
                        # Truncate and add ellipsis
                        anon_text = anon_text[:47] + "..."
                    annot = page.add_redact_annot(inst, text=anon_text)
                    # Apply redactions
                    page.apply_redactions()
            
            # 2. Face detection and blurring in images
            try:
                # Get page pixmap (rasterize the page)
                pix = page.get_pixmap()
                img_data = pix.tobytes()
                
                # Convert to PIL image
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                
                # Detect faces
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                faces = detect_faces(img_bytes.getvalue())
                
                # If faces detected, add white rectangles over them
                if faces and len(faces):
                    logger.info(f"Found {len(faces)} faces on page {page_idx+1}")
                    
                    for (x, y, w, h) in faces:
                        # Scale face coordinates to match the page dimensions
                        scale_x = page.rect.width / pix.width
                        scale_y = page.rect.height / pix.height
                        
                        # Create rectangle with some padding
                        padding = 5
                        face_rect = fitz.Rect(
                            x * scale_x - padding,
                            y * scale_y - padding,
                            (x + w) * scale_x + padding,
                            (y + h) * scale_y + padding
                        )
                        
                        # Add white rectangle with high opacity
                        page.draw_rect(face_rect, color=(1, 1, 1), fill=(1, 1, 1), opacity=0.9)
            
            except Exception as e:
                logger.error(f"Error processing faces on page {page_idx+1}: {e}")
                # Continue to next page even if this page fails
        
        # Save the anonymized PDF
        doc.save(temp_path)
        doc.close()
        
        return temp_path
    
    except Exception as e:
        logger.error(f"Error anonymizing PDF: {e}")
        return None





def detect_faces(image_bytes):
    """
    Detect faces in an image using OpenCV
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        List of face rectangles (x, y, width, height)
    """
    try:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if image was loaded properly
        if img is None or img.size == 0:
            logger.warning("Failed to decode image")
            return []
        
        # Load the pre-trained face detector model
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Verify the cascade loaded correctly
        if face_cascade.empty():
            logger.error("Failed to load face cascade classifier")
            return []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Handle the case where no faces are found
        if isinstance(faces, tuple) and len(faces) == 0:
            return []
        
        # Convert faces to a list if it's a numpy array
        # This avoids the "truth value of an array is ambiguous" error
        faces_list = []
        for (x, y, w, h) in faces:
            faces_list.append((int(x), int(y), int(w), int(h)))
        
        return faces_list
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        return []

# Safe image extraction from PDF function
def safely_extract_image(doc, xref):
    """
    Safely extract an image from a PDF document
    
    Args:
        doc: PyMuPDF document
        xref: Image reference
        
    Returns:
        Dictionary with image data or None if extraction failed
    """
    try:
        return doc.extract_image(xref)
    except Exception as e:
        logger.error(f"Failed to extract image (xref {xref}): {e}")
        return None
    
    

def blur_faces(image_bytes, faces):
    """
    Blur detected faces in an image
    
    Args:
        image_bytes: Binary image data
        faces: List of face rectangles from detect_faces
        
    Returns:
        Binary data of image with blurred faces
    """
    try:
        # Convert to PIL image for processing
        img = Image.open(io.BytesIO(image_bytes))
        
        # If we found faces, blur each one
        if len(faces) > 0:
            # Convert to numpy for OpenCV processing
            img_np = np.array(img)
            
            # Convert RGB to BGR for OpenCV if needed
            # Check if the array has 3 dimensions and the last dimension is 3 (RGB)
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # For each face, apply a heavy blur
            for (x, y, w, h) in faces:
                # Make sure coordinates are within image bounds
                x, y = max(0, x), max(0, y)
                w = min(w, img_np.shape[1] - x)
                h = min(h, img_np.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue  # Skip if dimensions are invalid
                
                # Get the face region
                face_region = img_np[y:y+h, x:x+w]
                
                # Apply a strong blur
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                
                # Replace the region with the blurred version
                img_np[y:y+h, x:x+w] = blurred_face
            
            # Convert back to RGB for PIL if needed
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            # Convert back to PIL Image
            img = Image.fromarray(img_np)
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        img_format = img.format or "PNG"
        
        # Some formats like JPEG2000 might not be supported by PIL's save
        # Fall back to PNG in those cases
        try:
            img.save(output_buffer, format=img_format)
        except (KeyError, IOError):
            img.save(output_buffer, format="PNG")
            
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error blurring faces: {e}")
        return image_bytes  # Return original if processing fails
    
    
    
def match_referees_by_keywords(article_keywords, referees):
    """
    Find the most suitable referees based on keyword matching
    Returns a list of referees sorted by relevance
    """
    article_keywords = [k.lower() for k in article_keywords]
    
    #print(article_keywords)
    
    referee_scores = []
    for referee in referees:
        if not referee.specialization:
            score = 0
        else:
            # Split referee specialization into individual keywords
            referee_keywords = [k.strip().lower() for k in referee.specialization.split(',')]
            
            #print(referee_keywords)
            
            # Count matches between article keywords and referee specialization
            matches = sum(1 for kw in article_keywords if any(kw in ref_kw or ref_kw in kw for ref_kw in referee_keywords))
            
            #print(matches)
            
            # Score is the percentage of article keywords that match referee specialization
            score = matches / len(article_keywords) if article_keywords else 0
        
        referee_scores.append((referee, score))
    
    # Sort by score in descending order
    #print(referee_scores)
    return sorted(referee_scores, key=lambda x: x[1], reverse=True)



def safe_operation(operation_func, error_message, default_return=None, *args, **kwargs):
    """
    Wrapper function to execute operations safely within a try-except block
    
    Args:
        operation_func: Function to execute
        error_message: Message to log if an error occurs
        default_return: Value to return if the operation fails
        *args, **kwargs: Arguments to pass to operation_func
        
    Returns:
        Result of operation_func or default_return if an error occurs
    """
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        return default_return
    
    
    
import fitz

def replace_image_legacy(page, xref, blurred_bytes, bbox):
    """
    Eski PyMuPDF sürümünde page.update_image() yoksa,
    yeni resmi belirlenen 'bbox' bölgesine insert_image ile ekliyoruz.
    'xref'li orijinal resim durabilir, ama görsel olarak kaplamış oluruz.
    """
    # Örneğin, x0,y0,x1,y1 = bbox
    page.insert_image(bbox, stream=blurred_bytes)

def anonymize_pdf_legacy(pdf_path, anonymize_map):
    """
    Eski PyMuPDF sürümleriyle çalışan bir örnek:
    1) Metin redaction
    2) Her sayfadaki resimleri -> detect_faces -> blur -> insert_image
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
    os.close(temp_fd)
    
    doc = fitz.open(pdf_path)
    
    # A) Metin anonimleştirme (redaction)
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for original, replacement in anonymize_map.items():
            text_instances = page.search_for(original)
            for inst in text_instances:
                annot = page.add_redact_annot(inst, text=replacement, fill=(1,1,1))
            page.apply_redactions(images=False)
    
    # B) Resimleri bul -> yüzleri bulanıklaştır -> insert_image
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        if not image_list:
            continue
        
        for img_info in image_list:
            xref = img_info[0]
            # Resmi çıkar
            img_dict = doc.extract_image(xref)
            if not img_dict:
                continue
            
            img_bytes = img_dict["image"]
            faces = detect_faces(img_bytes)
            if not faces:
                continue
            
            blurred_bytes = blur_faces(img_bytes, faces)
            if blurred_bytes != img_bytes:
                # Resmi ekleyeceğimiz dikdörtgeni bulmak gerekiyor
                # Her resmin bounding box'ına erişmek PyMuPDF eski sürümde
                # doğrudan kolay olmayabilir. Tek seçenek: sayfayı rasterize edin 
                # veya tahmini bir rect belirleyin. 
                # Örn. sayfanın tam boyutu:
                rect = page.rect  # Tam sayfa
                # Gerçek bounding box'ı bulacaksanız, 
                #   muhtemelen "img_info"dan width/height alıp 
                #   sayfadaki konumunu hesaplamanız gerek.
                
                replace_image_legacy(page, xref, blurred_bytes, rect)
    
    doc.save(temp_path)
    doc.close()
    return temp_path

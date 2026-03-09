with open('prepare.py', 'r') as f:
    lines = f.readlines()

out_lines = []
in_conflict = False
conflict_lines = []

for line in lines:
    if line.startswith('<<<<<<< Updated upstream'):
        in_conflict = True
        conflict_lines = []
    elif line.startswith('======='):
        if in_conflict:
            # wait for stashed changes
            pass
        else:
            out_lines.append(line)
    elif line.startswith('>>>>>>> Stashed changes'):
        in_conflict = False
        # Inject our manual resolution
        res = """            doc = doc_buffer[best_idx]
            doc_buffer[best_idx] = doc_buffer[-1]
            doc_buffer.pop()
            doc_lens[best_idx] = doc_lens[-1]
            doc_lens.pop()
            row_tensor[pos:pos + best_len] = torch.tensor(doc, dtype=torch.long)
            pos += best_len
        else:
            # No doc fits — crop shortest to fill remaining
            shortest_idx = min(range(len(doc_lens)), key=lambda i: doc_lens[i])
            doc = doc_buffer[shortest_idx]
            doc_buffer[shortest_idx] = doc_buffer[-1]
            doc_buffer.pop()
            doc_lens[shortest_idx] = doc_lens[-1]
            doc_lens.pop()
"""
        out_lines.append(res)
    else:
        if in_conflict:
            pass
        else:
            out_lines.append(line)

with open('prepare.py', 'w') as f:
    f.writelines(out_lines)

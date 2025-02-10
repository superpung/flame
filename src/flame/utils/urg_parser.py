import os
import re


def _get_bin_from_line(bin):
    bin = bin.strip().split()[0]
    if bin.startswith("["):
        bin = bin[1:-1]
    return bin


def _get_uncover_bin_from_line(bin):
    bins = []
    bins_list_match = re.match(r"\[(.*\[.*\])\s-\s(.*\[.*\])\]", bin)
    bin_list_match = re.match(r"\[(.*\[.*\])\]", bin)
    if bins_list_match:
        bins_list_start = bins_list_match.group(1)
        bins_list_end = bins_list_match.group(2)
        start_num = int(re.findall(r"\d+", bins_list_start)[0])
        end_num = int(re.findall(r"\d+", bins_list_end)[0])
        for i in range(start_num, end_num + 1):
            bins.append(bins_list_start.replace(str(start_num), str(i)))
    elif bin_list_match:
        bin_list = bin_list_match.group(1)
        bins.append(bin_list)
    else:
        bin = bin.strip().split()[0]
        bins.append(bin)
    return bins


def _get_bins_from_line(bin, cross_title_line):
    bins = bin.strip().split()
    titles = cross_title_line.strip().split()
    bin = f"{titles[0]}.{bins[0]}"
    for i in range(1, len(bins)):
        if bins[i].isdigit():
            break
        bin += f"+{titles[i]}.{bins[i]}"
    return bin


def get_cov_item(file_path):
    item = {}
    item["groups"] = {}
    item["covered"] = []
    item["covered_var"] = []
    item["covered_cross"] = []
    item["uncovered"] = []
    item["g_covered"] = []
    item["g_covered_var"] = []
    item["g_covered_cross"] = []
    item["g_uncovered"] = []
    if not os.path.exists(file_path):
        return item
    with open(file_path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("Group :") and lines[i + 1].strip() == "":
                current_group = lines[i].strip().split(" : ")[-1]
                current_group_instance = None
                item["groups"][current_group] = {
                    "instances": {},
                    "source_files": [],
                    "var": [],
                    "cross": [],
                    "covered_var": [],
                    "covered_cross": [],
                    "uncovered_var": []
                }
                i += 1
                current_group_source_files = []
                while i < len(lines) and not lines[i].startswith("----"):
                    if lines[i].startswith("Source File(s) :"):
                        i += 2
                        while lines[i].strip() != "":
                            current_group_source_files.append(lines[i].strip())
                            i += 1
                    if "Instances:" in lines[i]:
                        i += 3
                        while lines[i].strip() != "":
                            _instance = lines[i].strip().split()[-1]
                            item["groups"][current_group]["instances"][_instance] = {
                                "var": [],
                                "cross": [],
                                "covered_var": [],
                                "covered_cross": [],
                                "uncovered_var": []
                            }
                            i += 1
                    i += 1
                item["groups"][current_group]["source_files"] = current_group_source_files
            if lines[i].startswith("Summary for Group") and not lines[i].startswith("Summary for Group Instance"):
                _current_group = lines[i].strip().split()[-1]
                while i < len(lines) and not lines[i].startswith("----"):
                    if lines[i].startswith("VARIABLE"):
                        i += 1
                        while lines[i].strip() != "":
                            item["groups"][_current_group]["var"].append(lines[i].split()[0])
                            i += 1
                    if lines[i].startswith("CROSS"):
                        i += 1
                        while lines[i].strip() != "":
                            item["groups"][_current_group]["cross"].append(lines[i].split()[0])
                            i += 1
                    i += 1
            if lines[i].startswith("Summary for Group Instance"):
                _current_group_instance = lines[i].strip().split()[-1]
                if _current_group_instance not in item["groups"][current_group]["instances"]:
                    raise ValueError(f"Group Instance {_current_group_instance} not found in {current_group}")
                while i < len(lines) and not lines[i].startswith("----"):
                    if lines[i].startswith("VARIABLE"):
                        i += 1
                        while lines[i].strip() != "":
                            item["groups"][current_group]["instances"][_current_group_instance]["var"].append(lines[i].split()[0])
                            i += 1
                    if lines[i].startswith("CROSS"):
                        i += 1
                        while lines[i].strip() != "":
                            item["groups"][current_group]["instances"][_current_group_instance]["cross"].append(lines[i].split()[0])
                            i += 1
                    i += 1
            if lines[i].startswith("Group Instance :"):
                current_group_instance = lines[i].strip().split(" : ")[-1]
            if lines[i].startswith("Summary for Variable"):
                target = lines[i].strip().split()[-1]
                i += 1
                while i < len(lines) and not lines[i].startswith("----") and not lines[i].startswith("Group :"):
                    if lines[i].startswith("Bins"):
                        i += 3
                        while i < len(lines) and not lines[i].startswith("\n"):
                            name = target + "." + _get_bin_from_line(lines[i])
                            item["covered_var"].append(name)
                            if current_group_instance:
                                item["groups"][current_group]["instances"][current_group_instance]["covered_var"].append(name)
                            elif current_group:
                                item["groups"][current_group]["covered_var"].append(name)
                                item["g_covered_var"].append(current_group + "/" + name)
                            i += 1
                    if lines[i].startswith("Covered bins"):
                        i += 3
                        while not lines[i].startswith("\n"):
                            name = target + "." + _get_bin_from_line(lines[i])
                            item["covered_var"].append(name)
                            if current_group_instance:
                                item["groups"][current_group]["instances"][current_group_instance]["covered_var"].append(name)
                            elif current_group:
                                item["groups"][current_group]["covered_var"].append(name)
                                item["g_covered_var"].append(current_group + "/" + name)
                            i += 1
                    if lines[i].startswith("Uncovered bins"):
                        i += 3
                        while i < len(lines) and not lines[i].startswith("\n"):
                            uncovered_bins = [
                                target + "." + _
                                for _ in _get_uncover_bin_from_line(lines[i])
                            ]
                            item["uncovered"].extend(uncovered_bins)
                            if current_group_instance:
                                item["groups"][current_group]["instances"][current_group_instance]["uncovered_var"].extend(uncovered_bins)
                            elif current_group:
                                item["groups"][current_group]["uncovered_var"].extend(uncovered_bins)
                                item["g_uncovered"].extend([current_group + "/" + name for name in uncovered_bins])
                            i += 1
                    i += 1
                i -= 1
            if i < len(lines) and lines[i].startswith("Summary for Cross"):
                target = lines[i].strip().split()[-1]
                i += 1
                while i < len(lines) and not lines[i].startswith("----") and not lines[i].startswith("Group :"):
                    if lines[i].startswith("Bins"):
                        cross_title_line = lines[i + 2]
                        i += 3
                        while not lines[i].startswith("\n"):
                            name = (
                                target
                                + "."
                                + _get_bins_from_line(lines[i], cross_title_line)
                            )
                            item["covered_cross"].append(name)
                            if current_group_instance:
                                item["groups"][current_group]["instances"][current_group_instance]["covered_cross"].append(name)
                            elif current_group:
                                item["groups"][current_group]["covered_cross"].append(name)
                                item["g_covered_cross"].append(current_group + "/" + name)
                            i += 1
                    if lines[i].startswith("Covered bins"):
                        cross_title_line = lines[i + 2]
                        i += 3
                        while not lines[i].startswith("\n"):
                            name = (
                                target
                                + "."
                                + _get_bins_from_line(lines[i], cross_title_line)
                            )
                            item["covered_cross"].append(name)
                            if current_group_instance:
                                item["groups"][current_group]["instances"][current_group_instance]["covered_cross"].append(name)
                            elif current_group:
                                item["groups"][current_group]["covered_cross"].append(name)
                                item["g_covered_cross"].append(current_group + "/" + name)
                            i += 1
                    i += 1
                i -= 1
            i += 1
    item["covered_var"] = list(set(item["covered_var"]))
    item["covered_cross"] = list(set(item["covered_cross"]))
    item["covered"] = item["covered_var"] + item["covered_cross"]
    item["uncovered"] = list(set(item["uncovered"]))
    item["g_covered_var"] = list(set(item["g_covered_var"]))
    item["g_covered_cross"] = list(set(item["g_covered_cross"]))
    item["g_covered"] = item["g_covered_var"] + item["g_covered_cross"]
    item["g_uncovered"] = list(set(item["g_uncovered"]))
    return item


def get_all_covered(files):
    all_covered = []
    for file in files:
        item = get_cov_item(file)
        all_covered.extend(item["covered"])
        all_covered = list(set(all_covered))
    return all_covered

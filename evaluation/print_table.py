from tabulate import tabulate, SEPARATING_LINE

def extract_top_k(datas): return [datas[0], datas[2], datas[4]]

class PrintTable:
    def __init__(self, typet5_analysis, typegen_analysis, tiger_analysis):
        self.typet5_analysis = typet5_analysis
        self.typegen_analysis = typegen_analysis
        self.tiger_analysis = tiger_analysis

    def print_main_table(self):
        # Print Main Table
        main_typet5 = self.typet5_analysis.return_arg_ret_top_k()
        main_typegen = self.typegen_analysis.return_top_k()
        main_typegen_base = self.typegen_analysis.return_base_top_k()
        main_tiger = self.tiger_analysis.return_top_k()
        main_tiger_base = self.tiger_analysis.return_base_top_k()

        args_ret_data = [
            {
                "base_model": "TypeT5",
                "base_exact": extract_top_k(main_typet5["Exact"]["Total"][1]),
                "base_base": extract_top_k(main_typet5["Base"]["Total"][1]),
                "our_exact": extract_top_k(main_typet5["Exact"]["Total"][3]),
                "our_base": extract_top_k(main_typet5["Base"]["Total"][3]),
                "diff_percent": extract_top_k(main_typet5["Exact"]["Total"][5]),
                "diff_percent_base": extract_top_k(main_typet5["Base"]["Total"][5]),
                "separator_after": True,
            },
            {
                "base_model": "Tiger",
                "base_exact": extract_top_k(main_tiger[1]),
                "base_base": extract_top_k(main_tiger_base[1]),
                "our_exact": extract_top_k(main_tiger[3]),
                "our_base": extract_top_k(main_tiger_base[3]),
                "diff_percent": extract_top_k(main_tiger[5]),
                "diff_percent_base": extract_top_k(main_tiger_base[5]),
                "separator_after": True,
            },
            {
                "base_model": "TypeGen",
                "base_exact": extract_top_k(main_typegen[1]),
                "base_base": extract_top_k(main_typegen_base[1]),
                "our_exact": extract_top_k(main_typegen[3]),
                "our_base": extract_top_k(main_typegen_base[3]),
                "diff_percent": extract_top_k(main_typegen[5]),
                "diff_percent_base": extract_top_k(main_typegen_base[5]),
            }
        ]

        headers = [
            "Model", "Exact T1(%)", "Exact T3(%)", "Exact T5(%)",
            "Base T1(%)", "Base T3(%)", "Base T5(%)"
        ]
        table_data = []

        for item in args_ret_data:
            base_row = [
                item["base_model"],
                f'{item["base_exact"][0]:.1f}%',
                f'{item["base_exact"][1]:.1f}%',
                f'{item["base_exact"][2]:.1f}%',
                f'{item["base_base"][0]:.1f}%',
                f'{item["base_base"][1]:.1f}%',
                f'{item["base_base"][2]:.1f}%',
            ]
            table_data.append(base_row)

            our_row = ["+Ours"]
            for i in range(3):
                our_row.append(f'({item["diff_percent"][i]:+.1f}%) {item["our_exact"][i]:.1f}%')
            for i in range(3):
                our_row.append(f'({item["diff_percent_base"][i]:+.1f}%) {item["our_base"][i]:.1f}%')
            table_data.append(our_row)
            
            if item.get("separator_after"):
                table_data.append(SEPARATING_LINE)

        print(tabulate(table_data, headers=headers, tablefmt="simple", stralign="right"))

    def print_signature_table(self):
        signature_data = self.typet5_analysis.return_top_k()
        signature_base_data = self.typet5_analysis.return_base_top_k()

        typet5_datas = ["TypeT5"] + extract_top_k(signature_data[1]) + extract_top_k(signature_base_data[1])
        ours_datas = ["Ours"] + extract_top_k(signature_data[3]) + extract_top_k(signature_base_data[3])
        improved = ["Improve"] + extract_top_k(signature_data[5]) + extract_top_k(signature_base_data[5])

        datas = [
            typet5_datas,
            ours_datas,
            improved
        ]

        simple_headers = [
            "Model",
            "Exact Top-1",
            "Exact Top-3",
            "Exact Top-5",
            "Base Top-1",
            "Base Top-3",
            "Base Top-5"
        ]

        print(tabulate(datas, headers=simple_headers, tablefmt="simple", stralign="right"))


    def print_param_return_table(self):
        arg_return_typet5 = self.typet5_analysis.return_arg_ret_top_k()
        arg_return_typegen = self.typegen_analysis.return_arg_ret_top_k()
        arg_return_tiger = self.tiger_analysis.return_arg_ret_top_k()

        tyepet5_data = ["TypeT5"] + extract_top_k(arg_return_typet5["Exact"]["Args"][1]) + extract_top_k(arg_return_typet5["Exact"]["Returns"][1])
        typet5_ours_data = ["Ours"] + extract_top_k(arg_return_typet5["Exact"]["Args"][3]) + extract_top_k(arg_return_typet5["Exact"]["Returns"][3])
        typet5_improved = ["Improve"] + extract_top_k(arg_return_typet5["Exact"]["Args"][5]) + extract_top_k(arg_return_typet5["Exact"]["Returns"][5])

        typegen_data = ["TypeGen"] + extract_top_k(arg_return_typegen["Exact"]["Args"][1]) + extract_top_k(arg_return_typegen["Exact"]["Returns"][1])
        typegen_ours_data = ["Ours"] + extract_top_k(arg_return_typegen["Exact"]["Args"][3]) + extract_top_k(arg_return_typegen["Exact"]["Returns"][3])
        typegen_improved = ["Improve"] + extract_top_k(arg_return_typegen["Exact"]["Args"][5]) + extract_top_k(arg_return_typegen["Exact"]["Returns"][5])

        tiger_data = ["Tiger"] + extract_top_k(arg_return_tiger["Exact"]["Args"][1]) + extract_top_k(arg_return_tiger["Exact"]["Returns"][1])
        tiger_ours_data = ["Ours"] + extract_top_k(arg_return_tiger["Exact"]["Args"][3]) + extract_top_k(arg_return_tiger["Exact"]["Returns"][3])
        tiger_improved = ["Improve"] + extract_top_k(arg_return_tiger["Exact"]["Args"][5]) + extract_top_k(arg_return_tiger["Exact"]["Returns"][5])

        datas = [
            tyepet5_data,
            typet5_ours_data,
            typet5_improved,
            SEPARATING_LINE,
            tiger_data,
            tiger_ours_data,
            tiger_improved,
            SEPARATING_LINE,
            typegen_data,
            typegen_ours_data,
            typegen_improved
        ]

        simple_headers = [
            "Model",
            "Param Top-1",
            "Param Top-3",
            "Param Top-5",
            "Return Top-1",
            "Return Top-3",
            "Return Top-5"
        ]

        print(tabulate(datas, headers=simple_headers, tablefmt="simple", stralign="right"))

    def print_categorized_table(self):
        categorized_typegen = self.typegen_analysis.return_categorized_top_k()
        categorized_tiger = self.tiger_analysis.return_categorized_top_k()

        typegen_data = ["TypeGen"] + extract_top_k(categorized_typegen["Elem"][1]) + extract_top_k(categorized_typegen["Param"][1]) + extract_top_k(categorized_typegen["User"][1])
        typegen_ours_data = ["Ours"] + extract_top_k(categorized_typegen["Elem"][3]) + extract_top_k(categorized_typegen["Param"][3]) + extract_top_k(categorized_typegen["User"][3])
        typegen_improved = ["Improve"] + extract_top_k(categorized_typegen["Elem"][5]) + extract_top_k(categorized_typegen["Param"][5]) + extract_top_k(categorized_typegen["User"][5]) 

        tiger_data = ["Tiger"] + extract_top_k(categorized_tiger["Elem"][1]) + extract_top_k(categorized_tiger["Param"][1]) + extract_top_k(categorized_tiger["User"][1])
        tiger_ours_data = ["Ours"] + extract_top_k(categorized_tiger["Elem"][3]) + extract_top_k(categorized_tiger["Param"][3]) + extract_top_k(categorized_tiger["User"][3])
        tiger_improved = ["Improve"] + extract_top_k(categorized_tiger["Elem"][5]) + extract_top_k(categorized_tiger["Param"][5]) + extract_top_k(categorized_tiger["User"][5])

        datas = [
            tiger_data,
            tiger_ours_data,
            tiger_improved,
            SEPARATING_LINE,
            typegen_data,
            typegen_ours_data,
            typegen_improved
        ]

        simple_headers = [
            "Model",
            "Elem Top-1",
            "Elem Top-3",
            "Elem Top-5",
            "Param Top-1",
            "Param Top-3",
            "Param Top-5",
            "User Top-1",
            "User Top-3",
            "User Top-5"
        ]

        print(tabulate(datas, headers=simple_headers, tablefmt="simple", stralign="right"))
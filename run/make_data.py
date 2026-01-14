from Levenshtein import distance as levenshtein_distance
from Levenshtein import jaro_winkler
from typet5.type_check import parse_type_str

def normalize_type_str(type_str):
    return str(parse_type_str(type_str))

def levenstein_similarity(str1, str2):
    return 1 - levenshtein_distance(str1, str2) / max(len(str1), len(str2))

def smart_levenshtein(a, b):
    lev_dist = levenshtein_distance(a, b)
    len_penalty = abs(len(a) - len(b)) / max(len(a), len(b))
    base_score = 1 - lev_dist / max(len(a), len(b))
    return base_score * (1 - len_penalty)

def jaro_winkler_similarity(a, b):
    return jaro_winkler(a, b)

def reversed_jaro_winkler_similarity(a, b):
    return jaro_winkler(a[::-1], b[::-1])

def make_data(
  ctx,
  filename,
  classname,
  funcname,
  decorators,
  args,
  var,
  cand_ctx,
  cand_filename,
  cand_classname,
  cand_funcname,
  cand_decorators,
  cand_args,
  cand_var,
  is_return=False,
  oracle_type=None,
  is_constraint=True,
):
    datas = []

    if is_return:
        assert var == funcname and cand_var == cand_funcname, \
            f"Expected var and funcname to be the same for return type, but got {var} and {funcname}, {cand_var} and {cand_funcname}"

    leven_sim = smart_levenshtein(var, cand_var)
    jaro_sim = jaro_winkler_similarity(var, cand_var)
    rev_jaro_sim = reversed_jaro_winkler_similarity(var, cand_var)

    if is_return:
        input_condition = (leven_sim > 0.75) and (jaro_sim > 0.95 or rev_jaro_sim > 0.95)
    else:
        input_condition = (leven_sim > 0.9) and (jaro_sim > 0.9 or rev_jaro_sim > 0.9)

    # if is_return:
    #     print(leven_sim, jaro_sim, rev_jaro_sim)

    if input_condition or True:
        if classname == "" or cand_classname == "":
            leven_sim_class = -1
            jaro_sim_class = -1
            rev_jaro_sim_class = -1
        else:
            leven_sim_class = smart_levenshtein(classname, cand_classname)
            jaro_sim_class = jaro_winkler_similarity(classname, cand_classname)
            rev_jaro_sim_class = reversed_jaro_winkler_similarity(classname, cand_classname)

        leven_sim_func = smart_levenshtein(funcname, cand_funcname)
        jaro_sim_func = jaro_winkler_similarity(funcname, cand_funcname)
        rev_jaro_sim_func = reversed_jaro_winkler_similarity(funcname, cand_funcname)

        for usage_ctx, _ in ctx.items():
            for cand_usage_ctx, type_counter in cand_ctx.items():
                ctx_len = len(usage_ctx)
                cand_ctx_len = len(cand_usage_ctx)

                ctx_sim = 0
                if min(ctx_len, cand_ctx_len) > 0:
                    ctx_sim = round(len(usage_ctx.intersection(cand_usage_ctx)) / min(ctx_len, cand_ctx_len), 3)
                else:
                    ctx_sim = -1

                deco_sim = 0
                if len(decorators) > 0 and len(cand_decorators) > 0:
                    deco_sim = round(len(set(decorators).intersection(set(cand_decorators))) / min(len(decorators), len(cand_decorators)), 3)
                else:
                    deco_sim = -1

                is_plural = (var == cand_var + 's' or var == cand_var + 'es') \
                            or (cand_var == var + 's' or cand_var == var + 'es')
                max_count = max(type_counter.values())
                
                if max_count > 0 and ((not is_constraint) or (input_condition or ctx_sim > 0.5)):
                    try:
                        top_types = [normalize_type_str(key) for key, count in type_counter.items() if count == max_count]
                    except SyntaxError as e:
                        # print(f"Error parsing annotation: {e}")
                        continue

                    ctx_ratio = round(ctx_len / (ctx_len + cand_ctx_len), 2) if (ctx_len + cand_ctx_len) > 0 else -1
                    target_ctx_ratio = round(cand_ctx_len / (ctx_len + cand_ctx_len), 2) if (ctx_len + cand_ctx_len) > 0 else -1

                    data = [
                        leven_sim,
                        jaro_sim,
                        rev_jaro_sim,
                        is_plural,
                        ctx_sim,
                        ctx_len,
                        cand_ctx_len,
                        ctx_ratio,
                        target_ctx_ratio,
                        int(classname == ""),
                        int(cand_classname == ""),
                        int(filename == cand_filename),
                        leven_sim_class,
                        jaro_sim_class,
                        rev_jaro_sim_class,
                        leven_sim_func,
                        jaro_sim_func,
                        rev_jaro_sim_func,
                        int(funcname in cand_funcname),
                        int(cand_funcname in funcname),
                        deco_sim,
                        len(decorators),
                        len(cand_decorators),
                        int(len(args)),
                        int(len(cand_args)),
                        int(args == cand_args),
                        top_types
                    ]

                    # if is_return:
                    #     if oracle_type not in top_types:
                    #         print(funcname, cand_funcname)
                    #         print(f"Data: {data}")
                    #         print(f" | Oracle Type: {oracle_type} | Top Types: {top_types}")
                    #         input()

                    datas.append(data)

    return datas

            
/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <string.h>

#include "./regcomp.h"

#define OPERATOR    0200    /* Bitmask of all operators */
enum InstType_Internal
{
    //CHAR = 0177,
    START = 0200, /* Start, used for marker on stack */
    //RBRA = 0201,
    //LBRA = 0202,
    LBRA_NC = 0203, // non-capturing group
    //OR = 0204,
    CAT = 0205, /* Concatentation, implicit operator */
    STAR = 0206, /* Closure, * */
    STAR_LAZY = 0207,
    PLUS = 0210, /* a+ == aa* */
    PLUS_LAZY = 0211,
    QUEST = 0212,  /* a? == a|nothing, i.e. 0 or 1 a's */
    QUEST_LAZY = 0213,
    COUNTED = 0214, /* counted repeat a{2} a{3,5} */
    COUNTED_LAZY = 0215,
    //ANY = 0300,
    //ANYNL = 0301,
    NOP = 0302, /* No operation, internal use only */
    //BOL = 0303,
    //EOL = 0304,
    //CCLASS = 0305,
    //NCCLASS = 0306,
    //BOW = 0307,
    //NBOW = 0310,
    //END = 0377
};

static Reclass ccls_w(1); // = { 'a', 'z','A','Z','0','9','_','_' };
static Reclass ccls_W(8); // = { '\n','\n','a', 'z', 'A', 'Z', '0', '9', '_', '_' };
static Reclass ccls_s(2); // = { '\t', '\t', '\n', '\n', '\r', '\r', '\f', '\f', '\v', '\v', ' ', ' ' };
static Reclass ccls_S(16);// ccls_S is the same as ccls_s
static Reclass ccls_d(4); // = { '0', '9' };
static Reclass ccls_D(32);// = { '\n', '\n', '0', '9' };
//static Reclass ccls_W_Neg = { 0, 0x9, 0xB, 0x2F, 0x3A, 0x40, 0x5B, 0x5E, 0x60, 0x60, 0x7B, 0xFFFFFFFF };
//static Reclass ccls_S_Neg = { 0, 0x8, 0xE, 0x1F, 0x21, 0xFFFFFFFF };
//static Reclass ccls_D_Neg = { 0, 0x9, 0xB, 0x2F, 0x3A, 0xFFFFFFFF };

int Reprog::add_inst(int t)
{
    Reinst inst;
    inst.type = t;
    inst.u2.left_id = 0;
    inst.u1.right_id = 0;
    return add_inst(inst);
}

int Reprog::add_inst(Reinst inst)
{
    insts.push_back(inst);
    return (int)insts.size() - 1;
}

int Reprog::add_class(Reclass cls)
{
    classes.push_back(cls);
    return (int)classes.size()-1;
}

Reinst& Reprog::inst_at(int id)
{
    return insts[id];
}

Reclass& Reprog::class_at(int id)
{
    return classes[id];
}

void Reprog::set_start_inst(int id)
{
    startinst_id = id;
}

int Reprog::get_start_inst() const
{
    return startinst_id;
}

int Reprog::inst_count() const
{
    return (int)insts.size();
}

int Reprog::classes_count() const
{
    return (int)classes.size();
}

void Reprog::set_groups_count(int groups)
{
    num_capturing_groups = groups;
}

int Reprog::groups_count() const
{
    return num_capturing_groups;
}

const Reinst* Reprog::insts_data() const
{
    return insts.data();
}

const int* Reprog::starts_data() const
{
    return startinst_ids.data();
}

int Reprog::starts_count() const
{
    return (int)startinst_ids.size();
}


class RegParser
{
    Reprog& m_prog;

    const char32_t* exprp;
    bool lexdone;

    int id_ccls_w = -1, id_ccls_W = -1, id_ccls_s = -1, id_ccls_d = -1, id_ccls_D = -1;

    char32_t yy; /* last lex'd Char */
    int yyclass_id; /* last lex'd class */
    short yy_min_count;
    short yy_max_count;

    bool nextc(char32_t& c) // return "quoted" == backslash-escape prefix
    {
        if(lexdone)
        {
            c = 0;
            return true;
        }
        c = *exprp++;
        if(c == '\\')
        {
            c = *exprp++;
            return true;
        }
        if(c == 0)
            lexdone = true;
        return false;
    }

    int bldcclass()
    {
        int type = CCLASS;
        std::vector<char32_t> cls;
        int builtins = 0;

        /* look ahead for negation */
        /* SPECIAL CASE!!! negated classes don't match \n */
        char32_t c;
        int quoted = nextc(c);
        if(!quoted && c == '^')
        {
            type = NCCLASS;
            quoted = nextc(c);
            cls.push_back('\n');
            cls.push_back('\n');
        }

        /* parse class into a set of spans */
        int count_char = 0;
        while(true)
        {
            count_char++;
            if(c == 0)
            {
                // malformed '[]'
                return 0;
            }
            if(quoted)
            {
                switch(c)
                {
                case 'n':
                    c = '\n';
                    break;
                case 'r':
                    c = '\r';
                    break;
                case 't':
                    c = '\t';
                    break;
                case 'a':
                    c = 0x07;
                    break;
                case 'b':
                    c = 0x08;
                    break;
                case 'f':
                    c = 0x0C;
                    break;
                case 'w':
                    builtins |= ccls_w.builtins;
                    quoted = nextc(c);
                    continue;
                case 's':
                    builtins |= ccls_s.builtins;
                    quoted = nextc(c);
                    continue;
                case 'd':
                    builtins |= ccls_d.builtins;
                    quoted = nextc(c);
                    continue;
                case 'W':
                    builtins |= ccls_W.builtins;
                    quoted = nextc(c);
                    continue;
                case 'S':
                    builtins |= ccls_S.builtins;
                    quoted = nextc(c);
                    continue;
                case 'D':
                    builtins |= ccls_D.builtins;
                    quoted = nextc(c);
                    continue;
                }
            }
            if(!quoted && c == ']' && count_char>1)
                break;
            if(!quoted && c == '-')
            {
                if (cls.size() < 1)
                {
                    // malformed '[]'
                    return 0;
                }
                quoted = nextc(c);
                if ((!quoted && c == ']') || c == 0)
                {
                    // malformed '[]'
                    return 0;
                }
                cls[cls.size() - 1] = c;
            }
            else
            {
                cls.push_back(c);
                cls.push_back(c);
            }
            quoted = nextc(c);
        }

         /* sort on span start */
        for (int p = 0; p < cls.size(); p += 2)
            for (int np = p + 2; np < cls.size(); np+=2)
                if (cls[np] < cls[p])
                {
                    c = cls[np];
                    cls[np] = cls[p];
                    cls[p] = c;
                    c = cls[np+1];
                    cls[np+1] = cls[p+1];
                    cls[p+1] = c;

                }

        /* merge spans */
        Reclass yycls;
        yycls.builtins=builtins;
        int p = 0;
        if( cls.size()>=2 )
        {
            int np = 0;
            yycls.chrs += cls[p++];
            yycls.chrs += cls[p++];
            for (; p < cls.size(); p += 2)
            {
                /* overlapping or adjacent ranges? */
                if (cls[p] <= yycls.chrs[np + 1] + 1)
                {
                    if (cls[p + 1] >= yycls.chrs[np + 1])
                        yycls.chrs.replace(np + 1, 1, 1, cls[p + 1]); /* coalesce */
                }
                else
                {
                    np += 2;
                    yycls.chrs += cls[p];
                    yycls.chrs += cls[p+1];
                }
            }
        }
        yyclass_id = m_prog.add_class(yycls);
        return type;
    }

    int lex(int dot_type)
    {
        int quoted;
        quoted = nextc(yy);
        if(quoted)
        {
            if (yy == 0)
                return END;
            // treating all quoted numbers as Octal, since we are not supporting backreferences
            if (yy >= '0' && yy <= '7')
            {
                yy = yy - '0';
                char32_t c = *exprp++;
                while( c >= '0' && c <= '7' )
                {
                    yy = (yy << 3) | (c - '0');
                    c = *exprp++;
                }
                return CHAR;
            }
            else
            {
                switch (yy)
                {
                case 't':
                    yy = '\t';
                    break;
                case 'n':
                    yy = '\n';
                    break;
                case 'r':
                    yy = '\r';
                    break;
                case 'a':
                    yy = 0x07;
                    break;
                case 'f':
                    yy = 0x0C;
                    break;
                case '0':
                    yy = 0;
                    break;
                case 'x':
                {
                    char32_t a = *exprp++;
                    char32_t b = *exprp++;
                    yy = 0;
                    if (a >= '0' && a <= '9') yy += (a - '0') << 4;
                    else if (a > 'a' && a <= 'f') yy += (a - 'a' + 10) << 4;
                    else if (a > 'A' && a <= 'F') yy += (a - 'A' + 10) << 4;
                    if (b >= '0' && b <= '9') yy += b - '0';
                    else if (b > 'a' && b <= 'f') yy += b - 'a' + 10;
                    else if (b > 'A' && b <= 'F') yy += b - 'A' + 10;
                    break;
                }
                case 'w':
                {
                    if (id_ccls_w < 0)
                    {
                        yyclass_id = m_prog.add_class(ccls_w);
                        id_ccls_w = yyclass_id;
                    }
                    else yyclass_id = id_ccls_w;
                    return CCLASS;
                }
                case 'W':
                {
                    if (id_ccls_W < 0)
                    {
                        Reclass cls = ccls_w;
                        cls.chrs += '\n';
                        cls.chrs += '\n';
                        yyclass_id = m_prog.add_class(cls);
                        id_ccls_W = yyclass_id;
                    }
                    else yyclass_id = id_ccls_W;
                    return NCCLASS;
                }
                case 's':
                {
                    if (id_ccls_s < 0)
                    {
                        yyclass_id = m_prog.add_class(ccls_s);
                        id_ccls_s = yyclass_id;
                    }
                    else yyclass_id = id_ccls_s;
                    return CCLASS;
                }
                case 'S':
                {
                    if (id_ccls_s < 0)
                    {
                        yyclass_id = m_prog.add_class(ccls_s);
                        id_ccls_s = yyclass_id;
                    }
                    else yyclass_id = id_ccls_s;
                    return NCCLASS;
                }
                case 'd':
                {
                    if (id_ccls_d < 0)
                    {
                        yyclass_id = m_prog.add_class(ccls_d);
                        id_ccls_d = yyclass_id;
                    }
                    else yyclass_id = id_ccls_d;
                    return CCLASS;
                }
                case 'D':
                {
                    if (id_ccls_D < 0)
                    {
                        Reclass cls = ccls_d;
                        cls.chrs += '\n';
                        cls.chrs += '\n';
                        yyclass_id = m_prog.add_class(cls);
                        id_ccls_D = yyclass_id;
                    }
                    else yyclass_id = id_ccls_D;
                    return NCCLASS;
                }
                case 'b':
                    return BOW;
                case 'B':
                    return NBOW;
                case 'A':
                    return BOL;
                case 'Z':
                    return EOL;
                }
                return CHAR;
            }
        }

        switch(yy)
        {
        case 0:
            return END;
        case '*':
            if (*exprp == '?')
            {
                exprp++;
                return STAR_LAZY;
            }
            return STAR;
        case '?':
            if (*exprp == '?')
            {
                exprp++;
                return QUEST_LAZY;
            }
            return QUEST;
        case '+':
            if (*exprp == '?')
            {
                exprp++;
                return PLUS_LAZY;
            }
            return PLUS;
        case '{': // counted repitition
        {
            if (*exprp<'0' || *exprp>'9') break;
            const char32_t* exprp_backup = exprp; // in case '}' is not found
            char buff[8];
            buff[0] = 0;
            for (int i = 0; i < 7 && *exprp != '}' && *exprp != ',' && *exprp != 0; i++, exprp++)
            {
                buff[i] = *exprp;
                buff[i + 1] = 0;
            }
            if (*exprp != '}' && *exprp != ',')
            {
                exprp = exprp_backup;
                break;
            }
            sscanf(buff, "%hd", &yy_min_count);
            if (*exprp != ',')
                yy_max_count = yy_min_count;
            else
            {
                yy_max_count = -1;
                exprp++;
                buff[0] = 0;
                for (int i = 0; i < 7 && *exprp != '}' && *exprp != 0; i++, exprp++)
                {
                    buff[i] = *exprp;
                    buff[i + 1] = 0;
                }
                if (*exprp != '}')
                {
                    exprp = exprp_backup;
                    break;
                }
                if (buff[0] != 0)
                    sscanf(buff, "%hd", &yy_max_count);
            }
            exprp++;
            if (*exprp == '?')
            {
                exprp++;
                return COUNTED_LAZY;
            }
            return COUNTED;
        }
        case '|':
            return OR;
        case '.':
            return dot_type;
        case '(':
            if (*exprp == '?' && *(exprp + 1) == ':')  // non-capturing group
            {
                exprp += 2;
                return LBRA_NC;
            }
            return LBRA;
        case ')':
            return RBRA;
        case '^':
            return BOL;
        case '$':
            return EOL;
        case '[':
            return bldcclass();
        }
        return CHAR;
    }
public:
    struct Item
    {
        int t;
        union
        {
            char32_t yy;
            int yyclass_id;
            struct
            {
                short n;
                short m;
            } yycount;
        } d;
    };
    std::vector<Item> m_items;

    bool m_has_counted;

    RegParser(const char32_t* pattern, int dot_type, Reprog& prog) : m_prog(prog)
    {
        exprp = pattern;
        lexdone = false;
        m_has_counted = false;
        int token;
        while ((token = lex(dot_type)) != END)
        {
            Item item;
            item.t = token;
            if (token == CCLASS || token == NCCLASS)
                item.d.yyclass_id = yyclass_id;
            else if (token == COUNTED || token == COUNTED_LAZY)
            {
                item.d.yycount.n = yy_min_count;
                item.d.yycount.m = yy_max_count;
                m_has_counted = true;
            }
            else
                item.d.yy = yy;
            m_items.push_back(item);
        }
    }
};

class RegCompiler
{
    Reprog& m_prog;

    struct Node
    {
        int id_first;
        int id_last;
    };

    int cursubid;
    int pushsubid;
    std::vector<Node> andstack;

    struct Ator
    {
        int t;
        int subid;
    };

    std::vector<Ator> atorstack;

    bool lastwasand;
    int	nbra;

    inline void pushand(int f, int l)
    {
        andstack.push_back({ f, l });
    }

    inline Node popand(int op)
    {
        if( andstack.size() < 1 )
        {
            //missing operand for op
            int inst_id = m_prog.add_inst(NOP);
            pushand(inst_id, inst_id);
        }
        Node node = andstack[andstack.size() - 1];
        andstack.pop_back();
        return node;
    }

    inline void pushator(int t)
    {
        Ator ator;
        ator.t = t;
        ator.subid = pushsubid;
        atorstack.push_back(ator);
    }

    inline Ator popator()
    {
        Ator ator = atorstack[atorstack.size() - 1];
        atorstack.pop_back();
        return ator;
    }

    void evaluntil(int pri)
    {
        Node op1, op2;
        int id_inst1, id_inst2;
        while( pri == RBRA || atorstack[atorstack.size() - 1].t >= pri )
        {
            Ator ator = popator();
            switch(ator.t)
            {
            default:
                // unknown operator in evaluntil
                break;
            case LBRA:		/* must have been RBRA */
                op1 = popand('(');
                id_inst2 = m_prog.add_inst(RBRA);
                m_prog.inst_at(id_inst2).u1.subid = ator.subid;//subidstack[subidstack.size()-1];
                m_prog.inst_at(op1.id_last).u2.next_id = id_inst2;
                id_inst1 = m_prog.add_inst(LBRA);
                m_prog.inst_at(id_inst1).u1.subid = ator.subid;//subidstack[subidstack.size() - 1];
                m_prog.inst_at(id_inst1).u2.next_id = op1.id_first;
                pushand(id_inst1, id_inst2);
                return;
            case OR:
                op2 = popand('|');
                op1 = popand('|');
                id_inst2 = m_prog.add_inst(NOP);
                m_prog.inst_at(op2.id_last).u2.next_id = id_inst2;
                m_prog.inst_at(op1.id_last).u2.next_id = id_inst2;
                id_inst1 = m_prog.add_inst(OR);
                m_prog.inst_at(id_inst1).u1.right_id = op1.id_first;
                m_prog.inst_at(id_inst1).u2.left_id = op2.id_first;
                pushand(id_inst1, id_inst2);
                break;
            case CAT:
                op2 = popand(0);
                op1 = popand(0);
                m_prog.inst_at(op1.id_last).u2.next_id = op2.id_first;
                pushand(op1.id_first, op2.id_last);
                break;
            case STAR:
                op2 = popand('*');
                id_inst1 = m_prog.add_inst(OR);
                m_prog.inst_at(op2.id_last).u2.next_id = id_inst1;
                m_prog.inst_at(id_inst1).u1.right_id = op2.id_first;
                pushand(id_inst1, id_inst1);
                break;
            case STAR_LAZY:
                op2 = popand('*');
                id_inst1 = m_prog.add_inst(OR);
                id_inst2 = m_prog.add_inst(NOP);
                m_prog.inst_at(op2.id_last).u2.next_id = id_inst1;
                m_prog.inst_at(id_inst1).u2.left_id = op2.id_first;
                m_prog.inst_at(id_inst1).u1.right_id = id_inst2;
                pushand(id_inst1, id_inst2);
                break;
            case PLUS:
                op2 = popand('+');
                id_inst1 = m_prog.add_inst(OR);
                m_prog.inst_at(op2.id_last).u2.next_id = id_inst1;
                m_prog.inst_at(id_inst1).u1.right_id = op2.id_first;
                pushand(op2.id_first, id_inst1);
                break;
            case PLUS_LAZY:
                op2 = popand('+');
                id_inst1 = m_prog.add_inst(OR);
                id_inst2 = m_prog.add_inst(NOP);
                m_prog.inst_at(op2.id_last).u2.next_id = id_inst1;
                m_prog.inst_at(id_inst1).u2.left_id = op2.id_first;
                m_prog.inst_at(id_inst1).u1.right_id = id_inst2;
                pushand(op2.id_first, id_inst2);
                break;
            case QUEST:
                op2 = popand('?');
                id_inst1 = m_prog.add_inst(OR);
                id_inst2 = m_prog.add_inst(NOP);
                m_prog.inst_at(id_inst1).u2.left_id = id_inst2;
                m_prog.inst_at(id_inst1).u1.right_id = op2.id_first;
                m_prog.inst_at(op2.id_last).u2.next_id = id_inst2;
                pushand(id_inst1, id_inst2);
                break;
            case QUEST_LAZY:
                op2 = popand('?');
                id_inst1 = m_prog.add_inst(OR);
                id_inst2 = m_prog.add_inst(NOP);
                m_prog.inst_at(id_inst1).u2.left_id = op2.id_first;
                m_prog.inst_at(id_inst1).u1.right_id = id_inst2;
                m_prog.inst_at(op2.id_last).u2.next_id = id_inst2;
                pushand(id_inst1, id_inst2);
                break;
            }
        }
    }

    void Operator(int t)
    {
        if (t == RBRA && --nbra < 0)
            //unmatched right paren
            return;
        if (t == LBRA)
        {
            nbra++;
            if (lastwasand)
                Operator(CAT);
        }
        else
            evaluntil(t);
        if (t != RBRA)
            pushator(t);
        lastwasand = (
            t == STAR || t == QUEST || t == PLUS ||
            t == STAR_LAZY || t == QUEST_LAZY || t == PLUS_LAZY ||
            t == RBRA);
    }

    void Operand(int t)
    {
        if (lastwasand)
            Operator(CAT);	/* catenate is implicit */
        int inst_id = m_prog.add_inst(t);
        if (t == CCLASS || t == NCCLASS)
            m_prog.inst_at(inst_id).u1.cls_id = yyclass_id;
        else if (t == CHAR || t==BOL || t==EOL)
            m_prog.inst_at(inst_id).u1.c = yy;
        pushand(inst_id, inst_id);
        lastwasand = true;
    }

    char32_t yy;
    int yyclass_id;

    void expand_counted(const std::vector<RegParser::Item>& in, std::vector<RegParser::Item>& out)
    {
        std::vector<int> lbra_stack;
        int rep_start = -1;

        out.clear();
        for (int i = 0; i < in.size(); i++)
        {
            if (in[i].t != COUNTED && in[i].t != COUNTED_LAZY)
            {
                out.push_back(in[i]);
                if (in[i].t == LBRA || in[i].t == LBRA_NC)
                {
                    lbra_stack.push_back(i);
                    rep_start = -1;
                }
                else if (in[i].t == RBRA)
                {
                    rep_start = lbra_stack[lbra_stack.size() - 1];
                    lbra_stack.pop_back();
                }
                else if ((in[i].t & 0300) != OPERATOR)
                {
                    rep_start = i;
                }
            }
            else
            {
                if (rep_start < 0) // broken regex
                    return;

                RegParser::Item item = in[i];
                if (item.d.yycount.n <= 0)
                {
                    // need to erase
                    for (int j = 0; j < i - rep_start; j++)
                        out.pop_back();
                }
                else
                {
                    // repeat
                    for (int j = 1; j < item.d.yycount.n; j++)
                        for (int k = rep_start; k < i; k++)
                            out.push_back(in[k]);
                }

                // optional repeats
                if (item.d.yycount.m >= 0)
                {
                    for (int j = item.d.yycount.n; j < item.d.yycount.m; j++)
                    {
                        RegParser::Item o_item;
                        o_item.t = LBRA_NC;
                        o_item.d.yy = 0;
                        out.push_back(o_item);
                        for (int k = rep_start; k < i; k++)
                            out.push_back(in[k]);
                    }
                    for (int j = item.d.yycount.n; j < item.d.yycount.m; j++)
                    {
                        RegParser::Item o_item;
                        o_item.t = RBRA;
                        o_item.d.yy = 0;
                        out.push_back(o_item);
                        if (item.t == COUNTED)
                        {
                            o_item.t = QUEST;
                            out.push_back(o_item);
                        }
                        else
                        {
                            o_item.t = QUEST_LAZY;
                            out.push_back(o_item);
                        }
                    }
                }
                else // infinite repeat
                {
                    RegParser::Item o_item;
                    o_item.d.yy = 0;

                    if (item.d.yycount.n > 0) // put '+' after last repetition
                    {
                        if (item.t == COUNTED)
                        {
                            o_item.t = PLUS;
                            out.push_back(o_item);
                        }
                        else
                        {
                            o_item.t = PLUS_LAZY;
                            out.push_back(o_item);
                        }
                    }
                    else // copy it once then put '*'
                    {
                        for (int k = rep_start; k < i; k++)
                            out.push_back(in[k]);

                        if (item.t == COUNTED)
                        {
                            o_item.t = STAR;
                            out.push_back(o_item);
                        }
                        else
                        {
                            o_item.t = STAR_LAZY;
                            out.push_back(o_item);
                        }
                    }
                }
            }
        }
    }


public:
    RegCompiler(const char32_t* pattern, int dot_type, Reprog& prog) : m_prog(prog)
    {
        // Parse
        std::vector<RegParser::Item> items;
        {
            RegParser parser(pattern, dot_type, m_prog);

            // Expand counted repetitions
            if (parser.m_has_counted)
                expand_counted(parser.m_items, items);
            else
                items = parser.m_items;
        }


        cursubid = 0;
        pushsubid = 0;

        /* Start with a low priority operator to prime parser */
        pushator(START - 1);

        lastwasand = false;
        nbra = 0;

        for (int i = 0; i < (int)items.size(); i++)
        {
            RegParser::Item item = items[i];
            int token = item.t;
            if (token == CCLASS || token == NCCLASS)
                yyclass_id = item.d.yyclass_id;
            else
                yy = item.d.yy;

            if (token == LBRA)
            {
                ++cursubid;
                pushsubid = cursubid;
            }
            else if (token == LBRA_NC)
            {
                pushsubid = 0;
                token = LBRA;
            }

            if ((token & 0300) == OPERATOR)
                Operator(token);
            else
                Operand(token);
        }

        /* Close with a low priority operator */
        evaluntil(START);
        /* Force END */
        Operand(END);
        evaluntil(START);
        if (nbra)
            ; // "unmatched left paren";
        /* points to first and only operand */
        m_prog.set_start_inst(andstack[andstack.size() - 1].id_first);
        m_prog.optimize1();
        m_prog.optimize2();
        m_prog.set_groups_count(cursubid);
    }
};

Reprog* Reprog::create_from(const char32_t* pattern)
{
    Reprog* rtn = new Reprog;
    RegCompiler compiler(pattern, ANY, *rtn);
    //rtn->print();
    return rtn;
}

//Reprog regcompnl(const char32_t* pattern)
//{
//	RegCompiler compiler(pattern, ANYNL);
//	return compiler.m_prog;
//}

void Reprog::optimize1()
{
    // Treat non-capturing LBRAs/RBRAs as NOOP
    for (int i = 0; i < (int)insts.size(); i++)
    {
        if (insts[i].type == LBRA || insts[i].type == RBRA)
        {
            if (insts[i].u1.subid < 1)
            {
                insts[i].type = NOP;
            }
        }
    }

    // get rid of NOP chains
    for (int i=0; i < inst_count(); i++)
    {
        if( insts[i].type != NOP )
        {
            {
                int target_id = insts[i].u2.next_id;
                while(insts[target_id].type == NOP)
                    target_id = insts[target_id].u2.next_id;
                insts[i].u2.next_id = target_id;
            }
            if( insts[i].type == OR )
            {
                int target_id = insts[i].u1.right_id;
                while(insts[target_id].type == NOP)
                    target_id = insts[target_id].u2.next_id;
                insts[i].u1.right_id = target_id;
            }
        }
    }
    // skip NOPs from the beginning
    {
        int target_id = startinst_id;
        while( insts[target_id].type == NOP)
            target_id = insts[target_id].u2.next_id;
        startinst_id = target_id;
    }
    // actually remove the no-ops
    std::vector<int> id_map(inst_count());
    int j = 0; // compact the ops (non no-ops)
    for( int i = 0; i < inst_count(); i++)
    {
        id_map[i] = j;
        if( insts[i].type != NOP )
        {
            insts[j] = insts[i];
            j++;
        }
    }
    insts.resize(j);
    // fix up the ORs
    for( int i=0; i < inst_count(); i++)
    {
        {
            int target_id = insts[i].u2.next_id;
            insts[i].u2.next_id = id_map[target_id];
        }
        if( insts[i].type == OR )
        {
            int target_id = insts[i].u1.right_id;
            insts[i].u1.right_id = id_map[target_id];
        }
    }
    // set the new start id
    startinst_id = id_map[startinst_id];
}

// expand leading ORs to multiple startinst_ids
void Reprog::optimize2()
{
    startinst_ids.clear();
    std::vector<int> stack;
    stack.push_back(startinst_id);
    while(stack.size() > 0)
    {
        int id = *(stack.end() - 1);
        stack.pop_back();
        const Reinst& inst = insts[id];
        if(inst.type == OR)
        {
            stack.push_back(inst.u2.left_id);
            stack.push_back(inst.u1.right_id);
        }
        else
        {
            startinst_ids.push_back(id);
        }
    }
    startinst_ids.push_back(-1); // terminator mark
}

void Reprog::print()
{
    printf("Instructions:\n");
    for(int i = 0; i < insts.size(); i++)
    {
        const Reinst& inst = insts[i];
        printf("%d :", i);
        switch (inst.type)
        {
        default:
            printf("Unknown instruction: %d, nextid= %d", inst.type, inst.u2.next_id);
            break;
        case CHAR:
            if( inst.u1.c <=32 || inst.u1.c >=127 )
                printf("CHAR, c = '0x%02x', nextid= %d", (unsigned)inst.u1.c, inst.u2.next_id);
            else
                printf("CHAR, c = '%c', nextid= %d", inst.u1.c, inst.u2.next_id);
            break;
        case RBRA:
            printf("RBRA, subid= %d, nextid= %d", inst.u1.subid, inst.u2.next_id);
            break;
        case LBRA:
            printf("LBRA, subid= %d, nextid= %d", inst.u1.subid, inst.u2.next_id);
            break;
        case OR:
            printf("OR, rightid=%d, leftid=%d, nextid=%d", inst.u1.right_id, inst.u2.left_id, inst.u2.next_id);
            break;
        case STAR:
            printf("STAR, nextid= %d", inst.u2.next_id);
            break;
        case PLUS:
            printf("PLUS, nextid= %d", inst.u2.next_id);
            break;
        case QUEST:
            printf("QUEST, nextid= %d", inst.u2.next_id);
            break;
        case ANY:
            printf("ANY, nextid= %d", inst.u2.next_id);
            break;
        case ANYNL:
            printf("ANYNL, nextid= %d", inst.u2.next_id);
            break;
        case NOP:
            printf("NOP, nextid= %d", inst.u2.next_id);
            break;
        case BOL:
            printf("BOL, c = '%c', nextid= %d", inst.u1.c, inst.u2.next_id);
            break;
        case EOL:
            printf("EOL, c = '%c', nextid= %d", inst.u1.c, inst.u2.next_id);
            break;
        case CCLASS:
            printf("CCLASS, cls_id=%d , nextid= %d", inst.u1.cls_id, inst.u2.next_id);
            break;
        case NCCLASS:
            printf("NCCLASS, cls_id=%d , nextid= %d", inst.u1.cls_id, inst.u2.next_id);
            break;
        case BOW:
            printf("BOW, nextid= %d", inst.u2.next_id);
            break;
        case NBOW:
            printf("NBOW, nextid= %d", inst.u2.next_id);
            break;
        case END:
            printf("END");
            break;
        }
        printf("\n");
    }

    printf("startinst_id=%d\n", startinst_id);
    if( startinst_ids.size() > 0 )
    {
        printf("startinst_ids:");
        for (size_t i = 0; i < startinst_ids.size(); i++)
            printf(" %d", startinst_ids[i]);
        printf("\n");
    }

    int count = (int)classes.size();
    printf("\nClasses %d\n",count);
    for( int i = 0; i < count; i++ )
    {
        const Reclass& cls = classes[i];
        int len = (int)cls.chrs.size();
        printf("%2d: ", i);
        for( int j=0; j < len; j += 2 )
        {
            char32_t c1 = cls.chrs[j];
            char32_t c2 = cls.chrs[j+1];
            if( c1 <= 32 || c1 >= 127 || c2 <= 32 || c2 >= 127 )
                printf("0x%02x-0x%02x",(unsigned)c1,(unsigned)c2);
            else
                printf("%c-%c",(char)c1,(char)c2);
            if( (j+2) < len )
                printf(", ");
        }
        printf("\n");
        if( cls.builtins )
        {
            int mask = cls.builtins;
            printf("   builtins(x%02X):",(unsigned)mask);
            if( mask & 1 )
                printf(" \\w");
            if( mask & 2 )
                printf(" \\s");
            if( mask & 4 )
                printf(" \\d");
            if( mask & 8 )
                printf(" \\W");
            if( mask & 16 )
                printf(" \\S");
            if( mask & 32 )
                printf(" \\D");
        }
        printf("\n");
    }
    if( num_capturing_groups )
        printf("Number of capturing groups: %d\n", num_capturing_groups);
}
